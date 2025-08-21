import os
import requests
import ssl
import requests.adapters
import urllib3
# [추가] 재시도 로직을 위한 임포트
from urllib3.util.retry import Retry

import pandas as pd
from datetime import datetime, timedelta, date
import urllib.parse
import time
import sqlite3
import google.generativeai as genai
import io
from itertools import groupby
import logging
import json
from typing import List, Union, Set, Tuple, Dict, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# --- 0. 회사 프로필 및 분석 설정 ---

# AI가 분석 기준으로 삼을 회사 프로필 정의 (주요 고객 및 분야 명시 강화)
COMPANY_PROFILE = """
우리 회사는 가상현실(VR/AR/XR) 및 시뮬레이션 기술을 기반으로 하는 군사 및 경찰 훈련체계 전문 기업입니다.
핵심 기술:
1. 위치 및 동작 인식 기술 기반의 정밀한 공간 정합 (가상환경과 현실공간 매칭)
2. 몰입형 가상환경 구현 및 실시간 상호작용
주요 사업 분야 및 고객:
- 영상 모의 사격 훈련 시스템 및 과학화 사격장 구축 (군, 경찰, 예비군 포함)
- 가상 공수 강하(낙하산) 훈련 시뮬레이터
- 박격포/전차/항공기 등 군사 장비 운용 및 전술 훈련 시뮬레이터
- 소부대 전투(CQB) 및 대테러 훈련 시스템 (경찰특공대, 특전사 등)
- 예비군 과학화 훈련체계 구축 및 유지보수
- 장비 유지보수(MRO)를 위한 가상/증강현실 솔루션
"""

# 광범위 탐색용 키워드
BROAD_KEYWORDS = {"훈련", "체계", "시스템", "모의", "가상", "증강", "시뮬레이터", "시뮬레이션", "과학화", "교육", "연구개발", "성능개량"}

# --- 1. 보안 및 네트워크 설정 클래스 ---

# [수정] 재시도(Retry) 전략을 지원하도록 CustomHttpAdapter 수정
class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    # retry_strategy 인자 추가 및 kwargs 처리
    def __init__(self, ssl_context=None, retry_strategy=None, **kwargs):
        self.ssl_context = ssl_context
        # retry_strategy가 제공되면 max_retries 설정 (requests가 인식하는 방식)
        if retry_strategy:
            kwargs['max_retries'] = retry_strategy
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        # urllib3.PoolManager 사용
        self.poolmanager = urllib3.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context
        )


# --- 1. 키워드 관리 ---
KEYWORD_FILE = "keywords.txt"
# 상세 키워드
INITIAL_KEYWORDS = ["지뢰","드론","시뮬레이터","시뮬레이션","전차","유지보수","MRO","항공","가상현실","증강현실","훈련","VR","AR","MR","XR","대테러","소부대","CQB","특전사","경찰청","영상사격","모의사격","영상 모의","공수강하","박격포","예비군"]

def load_keywords(initial_keywords: list) -> set:
    if not os.path.exists(KEYWORD_FILE):
        save_keywords(set(initial_keywords))
        return set(initial_keywords)
    try:
        with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logging.error(f"Error loading keywords: {e}")
        return set(initial_keywords)

def save_keywords(keywords: set):
    try:
        with open(KEYWORD_FILE, 'w', encoding='utf-8') as f:
            for keyword in sorted(list(keywords)):
                f.write(keyword + '\n')
    except Exception as e:
        logging.error(f"Error saving keywords: {e}")

# --- 2. 유틸리티 및 API 클라이언트 ---

def safe_int(value):
    try:
        return int(float(value)) if value is not None and str(value).strip() != "" else None
    except (ValueError, TypeError):
        return None

def format_price(x):
    if pd.isna(x):
        return ""
    try:
        if isinstance(x, str):
             x = x.replace(',', '')
        if str(x).strip() == "":
             return ""
        return f"{int(float(x)):,}"
    except (ValueError, TypeError):
        return ""

def save_integrated_excel(data_frames: dict) -> bytes:
    """통합 엑셀 저장. AI 점수 조건부 서식, 자동 줄 바꿈 및 행 높이 자동 조절 기능 추가."""
    output = io.BytesIO()
    try:
        import xlsxwriter.utility

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- 1. 스타일 정의 ---
            # 헤더 스타일
            order_plan_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#FFF2CC', 'border': 1})
            main_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#D3D3D3', 'border': 1})
            ai_analysis_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#E2F0D9', 'border': 1})
            
            # 데이터 셀 스타일 (핵심: 자동 줄 바꿈 적용)
            data_cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })

            for sheet_name, df_data in data_frames.items():
                if df_data is None or df_data.empty: continue
                
                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet

                # --- 2. 헤더 작성 ---
                is_multi_index = isinstance(df_data.columns, pd.MultiIndex) and df_data.columns.nlevels == 2
                
                if is_multi_index:
                    # MultiIndex 헤더 처리 ('종합 현황 보고서')
                    col_idx = 0
                    level0_headers = df_data.columns.get_level_values(0)
                    level1_headers = df_data.columns.get_level_values(1)

                    def get_header_format(header_name):
                        if header_name == 'AI 분석 결과': return ai_analysis_header_format
                        return main_header_format

                    for header, group in groupby(level0_headers):
                        span = len(list(group))
                        current_format = get_header_format(header)
                        if span > 1:
                            worksheet.merge_range(0, col_idx, 0, col_idx + span - 1, str(header), current_format)
                        else:
                            worksheet.write(0, col_idx, str(header), current_format)
                        for i in range(span):
                            worksheet.write(1, col_idx + i, str(level1_headers[col_idx + i]), current_format)
                        col_idx += span
                    start_row = 2 # 데이터 시작 행
                else:
                    # 일반 헤더 처리 ('발주계획 현황' 등)
                    for col_num, value in enumerate(df_data.columns.values):
                        current_format = order_plan_header_format
                        if 'AI' in value: current_format = ai_analysis_header_format
                        worksheet.write(0, col_num, value, current_format)
                    start_row = 1 # 데이터 시작 행

                # --- 3. 컬럼 너비 계산 및 설정 ---
                col_widths = []
                data_for_calc = df_data.fillna("").values.tolist()
                headers = df_data.columns
                
                for i in range(len(headers)):
                    header_text = str(headers[i]) if not is_multi_index else f"{headers.get_level_values(0)[i]}\n{headers.get_level_values(1)[i]}"
                    header_len = max(len(s) for s in header_text.split('\n'))
                    
                    try:
                        data_max_len = max((len(str(row[i])) for row in data_for_calc), default=0)
                    except Exception:
                        data_max_len = 10
                        
                    # 최종 너비 결정 (분석 이유는 더 넓게)
                    is_reason_col = (is_multi_index and headers.get_level_values(1)[i] == '분석 이유') or \
                                    (not is_multi_index and headers[i] == 'AI 분석 이유')
                                    
                    width = min(80, max(15, header_len, data_max_len) + 2) if is_reason_col else min(60, max(10, header_len, data_max_len) + 2)
                    
                    worksheet.set_column(i, i, width)
                    col_widths.append(width)

                # --- 4. 데이터 작성 및 행 높이 자동 조절 ---
                # to_excel 대신 수동으로 데이터를 쓰면서 행 높이를 계산하고 서식을 적용
                for row_num, row_data in enumerate(data_for_calc, start=start_row):
                    max_lines = 1
                    for i, cell_text in enumerate(row_data):
                        if len(str(cell_text)) > 0 and col_widths[i] > 0:
                            # 간단한 휴리스틱으로 줄 수 계산 (글자 수 / (열 너비 / 보정계수))
                            # 보정계수 1.8은 한글/영문 혼용 환경에서 평균적으로 잘 동작하는 값
                            lines_needed = -(-len(str(cell_text)) // (col_widths[i] / 1.8))
                            if lines_needed > max_lines:
                                max_lines = lines_needed
                    
                    # 행 높이 설정 (기본 15pt * 줄 수)
                    row_height = 15 * max_lines
                    worksheet.set_row(row_num, min(400, row_height)) # 최대 높이 제한
                    
                    # 서식과 함께 데이터 행 작성
                    worksheet.write_row(row_num, 0, row_data, data_cell_format)

                # --- 5. 조건부 서식 적용 ---
                score_col_idx = -1
                score_col_name = '관련성 점수' if is_multi_index else 'AI 관련성 점수'
                cols = headers.get_level_values(1) if is_multi_index else headers
                
                try:
                    score_col_idx = list(cols).index(score_col_name)
                except ValueError:
                    score_col_idx = -1

                if score_col_idx != -1:
                    col_letter = xlsxwriter.utility.xl_col_to_name(score_col_idx)
                    cell_range = f'{col_letter}{start_row + 1}:{col_letter}{len(data_for_calc) + start_row}'
                    worksheet.conditional_format(cell_range, {
                        'type': '3_color_scale',
                        'min_value': 0, 'mid_value': 50, 'max_value': 100,
                        'min_color': "#F8696B", 'mid_color': "#FFEB84", 'max_color': "#63BE7B"
                    })

    except ImportError:
        logging.error("xlsxwriter module not found or initialization failed.")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df_data in data_frames.items():
                 if df_data is not None and not df_data.empty:
                    df_data.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.info("Fallback to openpyxl engine for Excel export. Formatting will be limited.")
    except Exception as e:
        logging.error(f"Error saving Excel file: {e}")
        raise

    return output.getvalue()

# --- 2. 수정된 NaraJangteoApiClient 클래스 (안정성 강화) ---
class NaraJangteoApiClient:
    # [수정] 재시도 로직 적용
    def __init__(self, service_key: str):
        if not service_key: raise ValueError("서비스 키는 필수입니다.")
        self.service_key = service_key
        self.base_url_std = "https://apis.data.go.kr/1230000/ao/PubDataOpnStdService"
        self.base_url_plan = "https://apis.data.go.kr/1230000/ao/OrderPlanSttusService"

        # 1. 보안 설정 (SSL Context)
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        # 서버와 호환성을 높이기 위해 암호화 방식을 설정합니다. (공공데이터포털 호환성 이슈 대응)
        ctx.set_ciphers('ALL:@SECLEVEL=1')
        
        # 2. [신규] 자동 재시도 전략 설정 (HTTP 500 및 네트워크 오류 대응)
        retry_strategy = Retry(
            total=3,  # 최대 3회 재시도
            status_forcelist=[500, 502, 503, 504], # 이 상태 코드에서 재시도
            allowed_methods=["HEAD", "GET", "OPTIONS"], # GET 요청에 대해서만 재시도
            backoff_factor=1 # 재시도 간격 (1초, 2초, 4초...)
        )

        # 3. 세션 설정 및 어댑터 연결
        self.session = requests.Session()
        
        # 보안 설정(ctx)과 재시도 전략(retry_strategy)을 모두 적용한 어댑터 생성
        adapter = CustomHttpAdapter(ssl_context=ctx, retry_strategy=retry_strategy)
        
        self.session.mount("https://", adapter)


    # [핵심 수정: 오류 처리 순서 교정 및 로깅 강화]
    def _make_request(self, base_url: str, endpoint: str, params: dict, log_list: list):
        
        decoded_key = "NOT_SET"
        response = None

        try:
            # 1. 서비스 키 디코딩 및 URL 구성
            try:
                decoded_key = urllib.parse.unquote(self.service_key)
            except Exception as e:
                log_list.append(f"⚠️ [내부오류] ({type(e).__name__}) 서비스 키 디코딩 실패: {e}.")
                return []

            url = f"{base_url}/{endpoint}?ServiceKey={decoded_key}"

            # 2. 나머지 파라미터 준비
            other_params = {
                'pageNo': 1,
                'numOfRows': 999,
                'type': 'json',
                **params
            }
            
            # 3. 요청 수행
            response = self.session.get(url, params=other_params, timeout=90)
            
            # 4. HTTP 상태 코드 확인
            # 재시도 로직이 모두 실패하면 최종적으로 HTTPError가 발생합니다.
            response.raise_for_status()

            # 5. 응답 본문 검증
            content = response.text.strip()
            if not content:
                log_list.append(f"ℹ️ [API통신] 응답 본문이 비어 있습니다 ({endpoint}). Status: {response.status_code}.")
                return []

            # 6. JSON 파싱 시도
            # requests.exceptions.JSONDecodeError 또는 ValueError/json.JSONDecodeError 발생 가능
            data = response.json()

        except requests.exceptions.SSLError as e:
            log_list.append(f"⚠️ [SSL오류] ({type(e).__name__}) SSL 통신 오류 발생 ({endpoint}): {e}")
            return []
        
        # *** HTTP 오류 처리 ***
        except requests.exceptions.HTTPError as e:
            log_list.append(f"⚠️ [HTTP오류] ({type(e).__name__}) 서버 에러 발생 ({endpoint}): {e}. (자동 재시도 실패)")
            if response:
                log_list.append(f"  - 서버 응답 샘플 (진단용): {response.text[:1000]}")
                if response.status_code >= 500:
                     log_list.append("  💡 [진단] HTTP 500/50x 에러는 서버 내부 문제입니다. 자동 재시도를 시도했으나 계속 실패했습니다. 서버 안정화가 필요합니다.")
            return []

        # *** JSON 디코딩 오류 처리 (순서 교정) ***
        # [중요] requests.exceptions.JSONDecodeError는 RequestException을 상속하므로, RequestException보다 먼저 처리해야 합니다.
        except (requests.exceptions.JSONDecodeError, json.JSONDecodeError, ValueError) as e:
            # 로그에서 발생한 "Expecting value..." 오류를 여기서 처리합니다.
            log_list.append(f"⚠️ [JSON파싱오류] ({type(e).__name__}) API 응답 형식이 JSON이 아닙니다 ({endpoint}). 오류: {e}")
            if response:
                content_sample = response.text[:1000]
                log_list.append(f"  - 서버 응답 샘플 (진단용): {content_sample}")
                if "<" in content_sample and ">" in content_sample:
                    log_list.append("  💡 [진단] 서버가 JSON 대신 XML/HTML 형식의 응답을 반환했습니다. API 서버의 일시적 문제일 수 있습니다.")
            return []
        
        # *** 네트워크 오류 처리 ***
        except requests.exceptions.RequestException as e:
            # 진짜 네트워크 에러 처리 (연결 실패, 타임아웃 등)
            log_list.append(f"⚠️ [네트워크오류] ({type(e).__name__}) 네트워크 연결 실패 또는 타임아웃 ({endpoint}): {e}. (자동 재시도 실패)")
            return []
        
        except Exception as e:
            # 기타 예상치 못한 모든 오류 처리
            log_list.append(f"⚠️ [예상치못한오류] ({type(e).__name__}) 요청 처리 중 알 수 없는 오류 발생 ({endpoint}): {e}")
            return []


        # 7. 성공적인 JSON 응답 처리 (API 로직)
        response_data = data.get('response', {})
        
        if not isinstance(response_data, dict):
             log_list.append(f"⚠️ [API구조오류] 예상치 못한 API 응답 구조입니다 ({endpoint}). 응답: {data}")
             return []

        header = response_data.get('header', {})
        result_code = header.get('resultCode', '00')

        if result_code != '00':
            log_list.append(f"⚠️ [API내부오류] API 오류 코드 수신 ({endpoint}) - 코드: {result_code}, 메시지: {header.get('resultMsg', '메시지 없음')}")
            return []

        body = response_data.get('body', {})

        if isinstance(body, list): return body
        if isinstance(body, dict):
            if body.get('totalCount', 0) == 0:
                 return []
            items = body.get('items', [])
            if isinstance(items, list):
                return items
        
        log_list.append(f"⚠️ [API구조오류] API 응답의 body 형식이 예상과 다릅니다 ({endpoint}). Body: {body}")
        return []

    # (나머지 API 호출 함수들은 변경사항 없음)
    def get_order_plans(self, year, log_list):
        years = [str(year)] if isinstance(year, (str, int)) else [str(y) for y in year]
        endpoints = {'물품': 'getOrderPlanSttusListThng', '용역': 'getOrderPlanSttusListServc', '공사': 'getOrderPlanSttusListConst'}
        all_plans = []

        for current_year in years:
            params = {'year': current_year}
            log_list.append(f"[{current_year}년도] 발주계획 조회 시작...")
            year_plans_count = 0
            for category, endpoint in endpoints.items():
                log_list.append(f"  - 카테고리: {category} 조회 중...")
                plans = self._make_request(self.base_url_plan, endpoint, params, log_list)
                if plans:
                    for plan in plans:
                        plan['category'] = category
                        plan['plan_year'] = current_year
                    all_plans.extend(plans)
                    year_plans_count += len(plans)
            log_list.append(f"[{current_year}년도] 총 {year_plans_count}건 수신.")

        log_list.append(f"발주계획 전체 총 {len(all_plans)}건 수신 완료.")
        return all_plans

    def get_pre_standard_specs(self, start_date, end_date, log_list):
        params = {'rgstBgnDt': start_date, 'rgstEndDt': end_date}
        return self._make_request(self.base_url_std, "getDataSetOpnStdPrdnmInfo", params, log_list)

    def get_bid_announcements(self, start_date, end_date, log_list):
        params = {'bidNtceBgnDt': start_date, 'bidNtceEndDt': end_date}
        return self._make_request(self.base_url_std, "getDataSetOpnStdBidPblancInfo", params, log_list)

    def get_successful_bid_info(self, start_date, end_date, log_list, bsns_div_cd):
        params = {'opengBgnDt': start_date, 'opengEndDt': end_date, 'bsnsDivCd': bsns_div_cd}
        return self._make_request(self.base_url_std, "getDataSetOpnStdScsbidInfo", params, log_list)

    def get_contract_info(self, start_date, end_date, log_list):
        params = {'cntrctCnclsBgnDate': start_date, 'cntrctCnclsEndDate': end_date}
        return self._make_request(self.base_url_std, "getDataSetOpnStdCntrctInfo", params, log_list)


# --- 3. 최적화된 데이터 처리 함수 (API 호출 단일화) 부터는 변경사항 없음 ---
# (이하 코드는 이전 답변의 코드와 동일합니다.)

def prepare_keywords(keywords: Set[str]) -> List[Tuple[str, str]]:
    """키워드를 검색에 최적화된 형태로 준비 (소문자화, 띄어쓰기 제거 버전)."""
    prepared = []
    for kw in keywords:
        kw_lower = kw.lower()
        kw_no_space = kw_lower.replace(" ", "")
        if kw_no_space:
             prepared.append((kw_lower, kw_no_space))
    return prepared

def fetch_api_data(fetch_function, params, log_list, log_prefix="") -> List[dict]:
    """API 호출 및 기본 처리를 담당합니다."""
    # 발주계획은 이미 API 클라이언트 내부에서 상세 로그를 기록함
    is_order_plan = log_prefix == "발주계획"

    if not is_order_plan:
        log_list.append(f"[{log_prefix}] API 데이터 요청 시작...")

    raw_data = fetch_function(log_list=log_list, **params)

    if not is_order_plan:
        if not raw_data:
            is_handled = False
            if log_list:
                 # 마지막 로그가 오류(⚠️), 정보(ℹ️), 또는 진단(💡)인 경우 이미 처리된 것으로 간주
                 is_handled = log_list[-1].startswith("⚠️") or log_list[-1].startswith("ℹ️") or log_list[-1].startswith("💡")
            if not is_handled:
                # 명시적인 오류나 정보 로그가 없는데 데이터가 없는 경우
                log_list.append("API에서 수신된 데이터가 없습니다.")
            return []
        log_list.append(f"API로부터 총 {len(raw_data)}건 수신.")

    return raw_data if raw_data else []

def filter_data(data: List[dict], prepared_keywords: List[Tuple[str, str]], search_fields: List[str]) -> List[dict]:
    """준비된 키워드와 다중 필드를 사용하여 데이터를 필터링합니다 (띄어쓰기 무시 포함)."""
    filtered_data = []

    for item in data:
        if not isinstance(item, dict): continue

        match_found = False
        # 지정된 모든 검색 필드(사업명, 기관명 등)에 대해 반복
        for field in search_fields:
            field_value = item.get(field)
            if not field_value: continue

            # 비교 대상 텍스트 준비
            target_text = str(field_value).lower()
            target_text_no_space = target_text.replace(" ", "")

            # 준비된 모든 키워드와 비교
            for kw_lower, kw_no_space in prepared_keywords:
                # 1. 원본 텍스트와 비교 (띄어쓰기 유지)
                if kw_lower in target_text:
                    match_found = True
                    break
                # 2. 띄어쓰기 제거된 텍스트와 비교
                if kw_no_space in target_text_no_space:
                     match_found = True
                     break
            if match_found:
                break

        if match_found:
            filtered_data.append(item)

    return filtered_data

# --- 4. 데이터베이스 ---

def setup_database():
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()

    # projects 테이블 생성
    cursor.execute("CREATE TABLE IF NOT EXISTS projects (bidNtceNo TEXT PRIMARY KEY, bidNtceNm TEXT, ntcelnsttNm TEXT, presmptPrce INTEGER, bid_status TEXT DEFAULT '공고', bidNtceDate TEXT, sucsfCorpNm TEXT, cntrctAmt INTEGER, cntrctDate TEXT)")

    # order_plans 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_year TEXT, category TEXT, dminsttNm TEXT, prdctNm TEXT, asignBdgtAmt INTEGER,
            orderInsttNm TEXT, orderPlanPrd TEXT, cntrctMthdNm TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderPlanPrd)
        )
    """)

    # ALTER TABLE로 기존 컬럼 및 신규 컬럼 추가
    def add_column_if_not_exists(table, column, definition):
        try:
            cols = [info[1] for info in cursor.execute(f"PRAGMA table_info({table})")]
            if column not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception as e:
            logging.warning(f"Error altering table {table}: {e}")

    # 기존 컬럼
    add_column_if_not_exists("projects", "prestandard_status", "TEXT")
    add_column_if_not_exists("projects", "prestandard_no", "TEXT")
    add_column_if_not_exists("projects", "prestandard_date", "TEXT")

    # AI 관련성 및 수집 방식 컬럼 추가
    for table in ["projects", "order_plans"]:
        add_column_if_not_exists(table, "relevance_score", "INTEGER")
        add_column_if_not_exists(table, "relevance_reason", "TEXT")
        add_column_if_not_exists(table, "collection_method", "TEXT")

    conn.commit(); conn.close()

def upsert_project_data(df, stage):
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()

    for _, r in df.iterrows():
        try:
            if stage == 'bid':
                # ON CONFLICT DO UPDATE (UPSERT) 사용 (SQLite 3.24.0 이상 필요)
                cursor.execute("""
                    INSERT INTO projects
                    (bidNtceNo, bidNtceNm, ntcelnsttNm, presmptPrce, bidNtceDate, prestandard_status, prestandard_no, prestandard_date, relevance_score, relevance_reason, collection_method)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(bidNtceNo) DO UPDATE SET
                        bidNtceNm=excluded.bidNtceNm,
                        ntcelnsttNm=excluded.ntcelnsttNm,
                        presmptPrce=excluded.presmptPrce,
                        bidNtceDate=excluded.bidNtceDate,
                        prestandard_status=excluded.prestandard_status,
                        prestandard_no=excluded.prestandard_no,
                        prestandard_date=excluded.prestandard_date,
                        relevance_score=excluded.relevance_score,
                        relevance_reason=excluded.relevance_reason,
                        collection_method=excluded.collection_method
                """, (
                    r.get('bidNtceNo'), r.get('bidNtceNm'), r.get('ntcelnsttNm'), safe_int(r.get('presmptPrce')), r.get('bidNtceDate'),
                    r.get('prestandard_status'), r.get('prestandard_no'), r.get('prestandard_date'),
                    safe_int(r.get('relevance_score')), r.get('relevance_reason'), r.get('collection_method')
                ))
            elif stage == 'successful_bid':
                cursor.execute("""
                    UPDATE projects SET bid_status='낙찰', sucsfCorpNm=?
                    WHERE bidNtceNo=? AND (bid_status='공고' OR bid_status IS NULL OR bid_status='')
                """, (r.get('sucsfCorpNm'), r.get('bidNtceNo')))
            elif stage == 'contract':
                cursor.execute("""
                    UPDATE projects SET bid_status='계약완료', sucsfCorpNm=?, cntrctAmt=?, cntrctDate=?
                    WHERE bidNtceNo=?
                """, (r.get('rprsntCorpNm'), safe_int(r.get('cntrctAmt')), r.get('cntrctCnclsDate'), r.get('bidNtceNo')))
        except Exception as e:
            logging.error(f"Error upserting project data (stage: {stage}): {e} - Data: {r.to_dict()}")
            continue

    conn.commit(); conn.close()

def upsert_order_plan_data(df, log_list):
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()

    for _, r in df.iterrows():
        try:
            # ON CONFLICT DO UPDATE (UPSERT) 사용
            cursor.execute("""
                INSERT INTO order_plans
                (plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderInsttNm, orderPlanPrd, cntrctMthdNm, relevance_score, relevance_reason, collection_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderPlanPrd) DO UPDATE SET
                    relevance_score=excluded.relevance_score,
                    relevance_reason=excluded.relevance_reason,
                    collection_method=excluded.collection_method,
                    created_at=CURRENT_TIMESTAMP -- 갱신 시간 업데이트
            """, (
                r.get('plan_year'), r.get('category'), r.get('dminsttNm'), r.get('prdctNm'), safe_int(r.get('asignBdgtAmt')),
                r.get('orderInsttNm'), r.get('orderPlanPrd'), r.get('cntrctMthdNm'),
                safe_int(r.get('relevance_score')), r.get('relevance_reason'), r.get('collection_method')
            ))
        except Exception as e:
            log_list.append(f"⚠️ 경고: 발주계획 DB 삽입 중 오류 발생: {e} - 데이터: {r.to_dict()}")
            continue

    conn.commit()
    log_list.append(f"발주계획 정보 DB 저장/업데이트 완료 (처리된 레코드: {len(df)}건).")
    conn.close()


# --- 5. AI 분석, 리스크 분석 및 보고서 ---

# AI 관련성 점수 계산 함수 - 평가 기준 강화
def calculate_ai_relevance(api_key, df, data_type, log_list):
    """Gemini를 사용하여 관련성 점수를 계산합니다 (Batch 처리)."""
    if df.empty:
        return df

    if not api_key:
         log_list.append("ℹ️ Gemini API 키가 없어 AI 관련성 분석을 생략합니다.")
         if 'relevance_score' not in df.columns:
             df['relevance_score'] = None
             df['relevance_reason'] = 'API 키 없음'
         return df

    # 데이터 타입별 필드 정의
    fields_map = {
        'order_plan': {'title': 'prdctNm', 'org': 'dminsttNm', 'budget': 'asignBdgtAmt', 'category': 'category'},
        'bid': {'title': 'bidNtceNm', 'org': 'ntcelnsttNm', 'budget': 'presmptPrce'}
    }

    if data_type not in fields_map:
        return df

    fields = fields_map[data_type]
    log_list.append(f"🤖 AI 관련성 분석 시작 ({data_type}, 총 {len(df)}건)...")

    # 프롬프트 정의 - [수정] SyntaxError 방지를 위해 안정적인 문자열 형식으로 변경
    PROMPT_TEMPLATE = (
        "당신은 조달 정보 분석 전문가입니다. 아래 정의된 [회사 프로필]을 바탕으로, 제공된 조달 사업 목록이 이 회사와 얼마나 관련성이 높은지 평가해주세요.\n\n"
        f"[회사 프로필]\n{COMPANY_PROFILE}\n\n"
        "[평가 기준 (100점 만점) - 중요]\n"
        "- 90~100점: 회사의 핵심 기술(XR, 시뮬레이션)과 주력 분야(사격, 강하, 박격포, CQB, 예비군 훈련, 경찰특공대 훈련, 과학화 훈련장)에 완벽히 부합하는 훈련체계 사업.\n"
        "- 70~89점: 주력 분야는 아니지만, 회사의 핵심 기술을 활용하여 수행 가능한 시뮬레이터/가상현실 관련 사업 또는 MRO. (예: 항공/함정 시뮬레이터)\n"
        "- 50~69점: 시뮬레이션 요소가 일부 포함되어 있으나, 회사의 핵심 기술과의 연관성이 다소 낮은 사업. (예: 단순 교육 연구)\n"
        "- 30~49점: 단순 장비 도입 또는 일반 IT 용역 사업.\n"
        "- 0~29점: 전혀 관련 없는 사업.\n\n"
        "[지시 사항]\n"
        "1. 아래 제공된 사업 목록(JSON 형식)을 분석하세요.\n"
        "2. 각 사업별로 평가 점수(score)와 간단한 근거(reason)를 제시하세요.\n"
        "3. 결과는 반드시 아래 형식의 JSON 리스트로만 출력해야 합니다. 다른 설명은 포함하지 마세요.\n"
        "   예시: [{\"index\": 1, \"score\": 85, \"reason\": \"XR 기술 활용 가능한 해양 시뮬레이터\"}, ...]\n\n"
        "[사업 목록]\n"
        "{data_placeholder}"
    )

    try:
        genai.configure(api_key=api_key)
        # 응답 형식을 JSON으로 강제 설정
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

        results = []
        BATCH_SIZE = 30 # API 호출 효율화를 위한 배치 크기

        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i+BATCH_SIZE]
            data_list = []

            # 배치 데이터를 JSON 리스트 형식으로 구성
            for index, row in batch_df.iterrows():
                item = {
                    "index": index,
                    "title": row.get(fields['title'], 'N/A'),
                    "organization": row.get(fields['org'], 'N/A'),
                    "budget": format_price(row.get(fields['budget']))
                }
                if 'category' in fields:
                    item["category"] = row.get(fields['category'], 'N/A')
                data_list.append(item)

            data_str = json.dumps(data_list, ensure_ascii=False)
            prompt = PROMPT_TEMPLATE.replace("{data_placeholder}", data_str)

            # API 호출 및 응답 처리
            try:
                response = model.generate_content(prompt)
                batch_results = json.loads(response.text)

                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    raise ValueError("응답이 JSON 리스트 형식이 아님")

                log_list.append(f"  - AI 분석 진행 중 ({min(i+BATCH_SIZE, len(df))}/{len(df)})...")
                time.sleep(0.5) # API Rate Limit 고려

            except (json.JSONDecodeError, ValueError, Exception) as e:
                log_list.append(f"⚠️ AI 응답 처리 오류 (배치 {i}~{i+BATCH_SIZE}): {e}. 해당 배치는 생략합니다.")
                # 오류 발생 시 해당 배치는 기본값으로 처리
                for index in batch_df.index:
                    results.append({"index": index, "score": -1, "reason": f"AI 분석 오류: {e}"})

        # 결과 데이터를 원본 데이터프레임에 병합
        if results:
            valid_results = [r for r in results if isinstance(r, dict) and 'index' in r]
            if valid_results:
                results_df = pd.DataFrame(valid_results).set_index('index')
                results_df['score'] = pd.to_numeric(results_df['score'], errors='coerce').fillna(-1).astype(int)

                # 원본 DF에 결과 컬럼 추가 (인덱스 기준으로 병합)
                df.loc[results_df.index, 'relevance_score'] = results_df['score']
                df.loc[results_df.index, 'relevance_reason'] = results_df['reason']

        log_list.append("✅ AI 관련성 분석 완료.")
        return df

    except Exception as e:
        log_list.append(f"⚠️ AI 관련성 분석 중 치명적 오류 발생: {e}")
        logging.exception(e)
        return df

def get_gemini_analysis(api_key, df, log_list):
    if df.empty: log_list.append("AI가 분석할 데이터가 없습니다."); return None

    if not api_key:
        log_list.append("ℹ️ Gemini API 키가 없어 전략 분석을 생략합니다.")
        return None

    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        log_list.append("Gemini API로 맞춤형 전략 분석 시작...")

        # 프롬프트 전달 전 금액 포맷팅 적용
        df_for_prompt = df.copy()
        if 'presmptPrce' in df_for_prompt.columns:
             df_for_prompt['presmptPrce'] = df_for_prompt['presmptPrce'].apply(format_price)
        if 'cntrctAmt' in df_for_prompt.columns:
             df_for_prompt['cntrctAmt'] = df_for_prompt['cntrctAmt'].apply(format_price)

        data_for_prompt_str = df_for_prompt[[c for c in ['relevance_score', 'bidNtceNm', 'bid_status', 'ntcelnsttNm', 'presmptPrce', 'cntrctAmt'] if c in df_for_prompt.columns]].head(30).to_string()

        # [수정] SyntaxError 방지를 위해 안정적인 문자열 형식으로 변경
        prompt = (
            "당신은 '위치 및 동작 인식 기술'을 기반으로 VR/MR/AR/XR 가상환경과 현실공간을 정밀하게 매칭(공간 정합)하는 기술을 보유한, 군/경 훈련 시뮬레이터 전문 기업의 '신사업 전략팀장'입니다. "
            "아래 제공된 나라장터 조달 데이터를 바탕으로 심층 분석 보고서를 작성해주세요. 데이터의 'relevance_score'는 우리 회사와의 관련성을 나타냅니다 (100점 만점). 이 점수가 높은 사업에 집중하여 분석해주세요.\n\n"
            f"[분석 데이터]\n{data_for_prompt_str}\n\n---\n"
            "## 군·경 훈련 시뮬레이션 사업 기회 분석 보고서\n\n"
            "### 1. 총평 및 핵심 동향\n"
            "(우리 회사의 XR 및 공간 정합 기술과 관련성이 높은 훈련 시뮬레이션, 가상현실, MRO 사업의 동향을 분석해주세요.)\n\n"
            "### 2. 주요 사업 심층 분석 (관련성 점수 상위 3~5개)\n"
            "(관련성 점수(relevance_score)가 가장 높은 프로젝트를 선정하여 아래 표 형식으로 분석해주세요. '분석 및 제언' 항목에는 **우리 회사의 핵심 기술인 '공간 정합', '위치/동작 인식' 기술을 적용할 수 있는 지점**을 중점적으로 작성해주세요.)\n\n"
            "| 사업명 | 발주기관 | 관련성 점수 | 추정가격/계약금액 | 분석 및 제언 (자사 기술 연계 방안) |\n"
            "|---|---|---|---|---|\n"
            "| (사업명) | (기관명) | (점수) | (금액) | (예: 이 사업은 CQB 훈련 시뮬레이터로, **현실 공간과 가상 훈련 시나리오를 정밀하게 매칭하는 우리 기술**이 핵심 경쟁력이 될 수 있음.) |\n\n"
            "### 3. 기술 연계 가능 키워드\n"
            "(데이터에서 식별된 키워드 중, 우리 회사의 기술과 연결될 수 있는 핵심 키워드를 5개 이상 선정하여 불렛 포인트로 나열해주세요.)\n\n"
            "### 4. 차기 사업 전략 제언\n"
            "(위 분석을 종합하여, 우리 회사가 다음 분기에 집중해야 할 사업 영역, 기술 고도화 방향 등에 대한 구체적인 실행 전략을 1~2가지 제언해주세요.)"
        )
        response = model.generate_content(prompt); log_list.append("Gemini 맞춤형 전략 분석 완료.")
        return response.text
    except Exception as e:
        log_list.append(f"⚠️ Gemini API 호출 중 오류 발생: {e}")
        return None

def expand_keywords_with_gemini(api_key, df, existing_keywords, log_list):
    if df.empty: log_list.append("키워드 확장을 위한 분석 데이터가 없습니다."); return set()

    if not api_key:
        log_list.append("ℹ️ Gemini API 키가 없어 키워드 확장을 생략합니다.")
        return set()

    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        log_list.append("Gemini API로 지능형 키워드 확장 시작...")
        project_titles = pd.concat([df[col] for col in ['bidNtceNm', 'cntrctNm', 'prdctClsfcNoNm', 'prdctNm'] if col in df.columns]).dropna().unique()
        project_titles_str = '\n- '.join(project_titles[:50])

        # [수정] SyntaxError 방지를 위해 안정적인 문자열 형식으로 변경
        prompt = (
            "당신은 군/경 훈련 시뮬레이터 전문 기업의 조달 정보 분석가입니다. 우리 회사는 '위치/동작 인식', 'XR 공간 정합' 기술을 보유하고 있습니다. "
            "아래는 현재 우리가 사용중인 검색 키워드와, 최근 조달 시스템에서 발견된 사업명/품명 목록입니다. "
            "이 정보를 바탕으로, 우리 회사의 기술과 관련성이 높으면서도 기존 키워드에 없는 **새로운 검색 키워드를 5~10개 추천**해주세요. "
            "결과는 다른 설명 없이, 쉼표(,)로 구분된 키워드 목록으로만 제공해주세요.\n\n"
            f"[기존 키워드]\n{', '.join(sorted(list(existing_keywords)))}\n\n"
            f"[최근 발견된 사업명/품명]\n- {project_titles_str}\n\n"
            "[추천 키워드]"
        )
        response = model.generate_content(prompt)
        new_keywords = {k.strip() for k in response.text.strip().split(',') if k.strip() and len(k.strip()) > 1}
        log_list.append(f"Gemini가 추천한 신규 키워드: {new_keywords}")
        return new_keywords
    except Exception as e:
        log_list.append(f"⚠️ Gemini 키워드 확장 중 오류 발생: {e}")
        return set()

def analyze_project_risk(df: pd.DataFrame) -> pd.DataFrame:
    # (이전 버전 코드 사용)
    ongoing_df = df[df['bid_status'].isin(['공고', '낙찰'])].copy()
    if ongoing_df.empty: return pd.DataFrame()

    ongoing_df['score'] = 0
    ongoing_df['risk_reason'] = ''
    ongoing_df['bidNtceDate_dt'] = pd.to_datetime(ongoing_df['bidNtceDate'], errors='coerce')
    current_time = datetime.now()

    for index, row in ongoing_df.iterrows():
        reasons = []
        score = 0

        if pd.notna(row['bidNtceDate_dt']) and row['bid_status'] == '공고':
            days_elapsed = (current_time - row['bidNtceDate_dt']).days
            if days_elapsed > 30:
                score += 5
                reasons.append(f'공고 후 {days_elapsed}일 경과')

        if row.get('prestandard_status') == '해당 없음':
            score += 3
            reasons.append('사전규격 미공개')

        price = row.get('presmptPrce')
        if pd.notna(price) and isinstance(price, (int, float)) and 0 < price < 50000000:
            score += 2
            reasons.append('소규모 사업 (5천만원 미만)')

        ongoing_df.loc[index, 'score'] = score
        ongoing_df.loc[index, 'risk_reason'] = ', '.join(reasons)

    ongoing_df['risk_level'] = ongoing_df['score'].apply(lambda s: '높음' if s >= 7 else ('보통' if s >= 4 else '낮음'))

    risk_table = ongoing_df[['bidNtceNm', 'ntcelnsttNm', 'bid_status', 'risk_level', 'risk_reason']]
    return risk_table.rename(columns={'bidNtceNm':'사업명','ntcelnsttNm':'발주기관','bid_status':'진행 상태','risk_level':'리스크 등급','risk_reason':'주요 리스크'}).sort_values(by='리스크 등급',key=lambda x:x.map({'높음':0,'보통':1,'낮음':2}))


# 보고서 생성 함수
def create_report_data(db_path, log_list, min_relevance_score=0):
    log_list.append(f"DB에서 최종 데이터 조회 중 (최소 관련성 점수: {min_relevance_score})...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor() # 커서 생성
    report_data = {}

    try:
        # 1. 프로젝트(입찰공고 이후) 데이터 처리
        try:
            # 필터링 전 전체 평가 완료된 개수 확인용 쿼리 (AI 분석 오류 제외)
            total_count_query = "SELECT COUNT(*) FROM projects WHERE relevance_score IS NOT NULL AND relevance_score >= 0"
            cursor.execute(total_count_query)
            result = cursor.fetchone()
            total_projects_scored = result[0] if result else 0

            # 기존 데이터 로드 쿼리 (필터링 적용)
            query = f"""
                SELECT * FROM projects
                WHERE relevance_score >= ?
                ORDER BY relevance_score DESC, bidNtceDate DESC
            """
            all_projects_df = pd.read_sql_query(query, conn, params=(min_relevance_score,))

            # 진단 로그 추가
            log_list.append(f"📊 DB 프로젝트 현황: 총 {total_projects_scored}건 평가됨 -> {len(all_projects_df)}건이 임계값({min_relevance_score}점) 통과.")

        except pd.errors.DatabaseError as e:
            log_list.append(f"⚠️ 프로젝트 DB 조회 중 오류: {e}")
            all_projects_df = pd.DataFrame()


        if not all_projects_df.empty:
            flat_df = all_projects_df.copy()

            # 분석용 원본 데이터 저장 (금액: 숫자형)
            report_data["flat"] = flat_df.copy()

            # 보고서용 데이터 포맷팅
            for col in ['prestandard_date','bidNtceDate','cntrctDate']:
                if col in flat_df.columns:
                    flat_df[col] = pd.to_datetime(flat_df[col],errors='coerce').dt.strftime('%Y-%m-%d')

            for col in ['presmptPrce','cntrctAmt']:
                if col in flat_df.columns:
                    flat_df[col] = flat_df[col].apply(format_price)

            # 구조화된 데이터프레임 (MultiIndex) 생성 - AI 분석 결과 및 탐지 방식 추가
            structured_columns = {
                ('AI 분석 결과','관련성 점수'):flat_df.get('relevance_score'),
                ('AI 분석 결과','분석 이유'):flat_df.get('relevance_reason'),
                ('프로젝트 개요','사업명'):flat_df.get('bidNtceNm'),
                ('프로젝트 개요','발주기관'):flat_df.get('ntcelnsttNm'),
                ('진행 현황','종합 상태'):flat_df.get('bid_status'),
                ('진행 현황','낙찰/계약사'):flat_df.get('sucsfCorpNm'),
                ('입찰 공고 정보','공고일'):flat_df.get('bidNtceDate'),
                ('입찰 공고 정보','추정가격'):flat_df.get('presmptPrce'),
                ('계약 체결 정보','계약금액'):flat_df.get('cntrctAmt'),
                ('참조 정보','탐지 방식'):flat_df.get('collection_method'),
            }
            report_data["structured"] = pd.DataFrame(structured_columns)
            log_list.append("프로젝트 현황 보고서 데이터 생성 완료.")

        # 2. 발주계획 데이터 처리
        try:
            # 필터링 전 전체 평가 완료된 개수 확인용 쿼리 (중복 제거 고려, AI 분석 오류 제외)
            total_count_plan_query = """
                SELECT COUNT(*) FROM (
                    SELECT 1 FROM order_plans
                    WHERE relevance_score IS NOT NULL AND relevance_score >= 0
                    GROUP BY plan_year, category, dminsttNm, prdctNm
                )
            """
            cursor.execute(total_count_plan_query)
            result_plan = cursor.fetchone()
            total_plans_scored = result_plan[0] if result_plan else 0

            # 기존 데이터 로드 쿼리 (필터링 적용)
            query_plan = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER(PARTITION BY plan_year, category, dminsttNm, prdctNm ORDER BY created_at DESC) as rn
                    FROM order_plans
                    WHERE relevance_score >= ?
                ) WHERE rn = 1
                ORDER BY relevance_score DESC, asignBdgtAmt DESC
            """
            all_order_plans_df = pd.read_sql_query(query_plan, conn, params=(min_relevance_score,))

            # 진단 로그 추가
            log_list.append(f"📊 DB 발주계획 현황: 총 {total_plans_scored}건 평가됨 -> {len(all_order_plans_df)}건이 임계값({min_relevance_score}점) 통과.")

        except pd.errors.DatabaseError as e:
             log_list.append(f"⚠️ 발주계획 DB 조회 중 오류: {e}")
             all_order_plans_df = pd.DataFrame()

        if not all_order_plans_df.empty:
            order_plan_df = all_order_plans_df.copy()

            # 금액 포맷팅
            order_plan_df['asignBdgtAmt_formatted'] = order_plan_df['asignBdgtAmt'].apply(format_price)

            # 보고서용 컬럼명 변경 및 선택 - AI 점수 및 근거 추가
            order_plan_report_df = order_plan_df[[
                'relevance_score', 'relevance_reason', 'plan_year', 'category', 'dminsttNm', 'prdctNm', 'asignBdgtAmt_formatted', 'orderPlanPrd', 'collection_method'
            ]].rename(columns={
                'relevance_score': 'AI 관련성 점수',
                'relevance_reason': 'AI 분석 이유',
                'plan_year': '년도',
                'category': '구분',
                'dminsttNm': '수요기관명',
                'prdctNm': '품명 (사업명)',
                'asignBdgtAmt_formatted': '배정예산액',
                'orderPlanPrd': '발주예정시기',
                'collection_method': '탐지 방식'
            })

            report_data["order_plan"] = order_plan_report_df
            log_list.append("발주계획 현황 보고서 데이터 생성 완료.")

        if not report_data:
             # 안내 메시지 강화
             log_list.append(f"❌ 설정된 조건(점수 {min_relevance_score}점 이상)에 해당하는 데이터가 없습니다. 임계값을 낮추거나 기간을 조정해보세요.")
             return None

        return report_data

    except Exception as e:
        log_list.append(f"⚠️ 보고서 데이터 생성 중 예상치 못한 오류 발생: {e}")
        logging.exception(e)
        return None
    finally:
        conn.close()

# --- 6. 메인 실행 함수 및 파이프라인 (최적화 적용) ---

# 하이브리드 데이터 수집 및 분석 함수 (최적화 적용)
def collect_and_analyze(fetch_function, params, detailed_keywords: Set[str], broad_keywords: Set[str], search_fields: Union[str, List[str]], gemini_key, log_list, log_prefix, data_type):
    """최적화된 파이프라인으로 데이터 수집 및 분석을 수행합니다 (API 호출 단일화)."""

    # search_fields를 항상 리스트로 처리
    if isinstance(search_fields, str):
        search_fields_list = [search_fields]
    else:
        search_fields_list = search_fields

    # 1. 데이터 수집 (API 호출 단 1회)
    log_list.append(f"\n--- [{log_prefix}] 1단계: 통합 데이터 수집 시작 ---")

    # API 호출 (기간 내 모든 데이터 확보)
    raw_data = fetch_api_data(fetch_function, params, log_list, log_prefix)

    if not raw_data:
        return pd.DataFrame()

    # 광범위 키워드로 1차 필터링 (관련 가능성이 있는 데이터만 선별)
    log_list.append(f"광범위 키워드 필터링 시작 (필드: {', '.join(search_fields_list)})...")
    prepared_broad_keywords = prepare_keywords(broad_keywords)
    broad_data = filter_data(raw_data, prepared_broad_keywords, search_fields_list)
    log_list.append(f"광범위 키워드 필터링 후 {len(broad_data)}건 확보.")

    if not broad_data:
        return pd.DataFrame()

    # 2. 상세 키워드 매칭 (Score 100점)
    log_list.append(f"\n--- [{log_prefix}] 2단계: 상세 키워드 매칭 (Score 100) ---")

    # 상세 키워드 준비
    prepared_detailed_keywords = prepare_keywords(detailed_keywords)

    # 확보된 데이터 내에서 상세 키워드 매칭 수행
    keyword_hits_data = filter_data(broad_data, prepared_detailed_keywords, search_fields_list)
    log_list.append(f"상세 키워드 매칭 결과: {len(keyword_hits_data)}건.")

    df_keyword = pd.DataFrame(keyword_hits_data)
    if not df_keyword.empty:
        df_keyword['relevance_score'] = 100
        df_keyword['relevance_reason'] = '정확한 키워드/기관명 매칭 (강화됨)'
        df_keyword['collection_method'] = 'Keyword'

    # 3. AI 분석 대상 선별 (상세 키워드 매칭 제외)
    # 메모리 주소(id)를 사용하여 식별 (가장 확실하고 빠름)
    keyword_data_ids = {id(item) for item in keyword_hits_data}
    ai_candidate_data = [item for item in broad_data if id(item) not in keyword_data_ids]

    df_ai_candidates = pd.DataFrame(ai_candidate_data)

    # 4. AI 관련성 분석
    if df_ai_candidates.empty:
        log_list.append(f"\n--- [{log_prefix}] 3단계: AI 분석 ---")
        log_list.append("모든 데이터가 상세 키워드에 매칭되어 AI 분석을 생략합니다.")
        df_analyzed = pd.DataFrame()
    else:
        log_list.append(f"\n--- [{log_prefix}] 3단계: AI 관련성 분석 시작 (대상: {len(df_ai_candidates)}건) ---")
        # AI 분석 수행 (calculate_ai_relevance 함수 사용)
        df_analyzed = calculate_ai_relevance(gemini_key, df_ai_candidates, data_type, log_list)

        # collection_method 설정
        if 'relevance_score' in df_analyzed.columns:
             # AI 분석 오류(-1)가 아닌 경우에만 AI_Broad로 표시
            df_analyzed['collection_method'] = df_analyzed.apply(lambda row: 'AI_Broad' if row.get('relevance_score', -1) != -1 else 'Broad_Error', axis=1)

    # 5. 결과 통합
    final_df = pd.concat([df_keyword, df_analyzed], ignore_index=True)

    # 중복 제거 (최종 확인 - 고유키 사용)
    if data_type == 'bid' and 'bidNtceNo' in final_df.columns:
         final_df = final_df.drop_duplicates(subset=['bidNtceNo'])

    log_list.append(f"✅ [{log_prefix}] 최종 결과: 총 {len(final_df)}건 수집 및 분석 완료.")
    return final_df


# 메인 실행 함수 - 최적화된 파이프라인 적용
def run_analysis(search_keywords: set, client: NaraJangteoApiClient, gemini_key: str, start_date: date, end_date: date, auto_expand_keywords: bool = True, min_relevance_score: int = 60):
    log = []; all_found_data = {}

    # 실행 시점 기록 및 날짜 설정
    execution_time = datetime.now()
    fmt_date = '%Y%m%d'; fmt_datetime = '%Y%m%d%H%M'
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt_limit = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
    end_dt = min(execution_time, end_dt_limit)

    start_date_str = start_dt.strftime(fmt_date); end_date_str = end_dt.strftime(fmt_date)
    start_datetime_str = start_dt.strftime(fmt_datetime); end_datetime_str = end_dt.strftime(fmt_datetime)

    log.append(f"💡 분석 시작: 검색 기간 {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    if end_dt != end_dt_limit:
        # 시간 표기 개선 (KST 명시 시도)
        try:
            import pytz
            kst = pytz.timezone('Asia/Seoul')
            current_time_kst = datetime.now(kst).strftime('%H:%M')
            time_info = f"(KST {current_time_kst})"
        except ImportError:
             # pytz가 없으면 서버 시간 사용
            current_time_kst = end_dt.strftime('%H:%M')
            time_info = f"({current_time_kst})"
        
        log.append(f"ℹ️ 참고: 종료 시각이 현재 시각{time_info} 기준으로 조정되었습니다.")

    # --- 데이터 수집 및 처리 파이프라인 (최적화된 하이브리드 방식 적용) ---

    # 1. 발주계획
    log.append("\n========== 1. 발주계획 정보 수집 및 분석 ==========")
    # 검색 기간 외에 미래년도(현재 기준 다음해)까지 조회하는 것이 일반적이므로 범위 확장 고려
    current_year = datetime.now().year
    target_years = list(set(range(start_date.year, max(end_date.year, current_year + 1) + 1)))
    
    order_plan_params = {'year': target_years}

    # 검색 필드: 품명(prdctNm) + 수요기관명(dminsttNm)
    all_found_data['order_plan'] = collect_and_analyze(
        client.get_order_plans, order_plan_params, search_keywords, BROAD_KEYWORDS,
        search_fields=['prdctNm', 'dminsttNm'],
        gemini_key=gemini_key, log_list=log, log_prefix="발주계획", data_type="order_plan"
    )
    # DB 저장
    if not all_found_data['order_plan'].empty:
        upsert_order_plan_data(all_found_data['order_plan'], log)

    # 2. 사전규격 (참조용, 상세 키워드만 사용)
    log.append("\n========== 2. 사전규격 정보 수집 (참조용) ==========")
    pre_standard_params = {'start_date': start_date_str, 'end_date': end_date_str}

    # API 호출
    pre_std_raw = fetch_api_data(client.get_pre_standard_specs, pre_standard_params, log, log_prefix="사전규격")
    # 필터링 (상세 키워드만 사용)
    prepared_detailed = prepare_keywords(search_keywords)

    # pre_std_raw가 비어있지 않을 경우에만 필터링 수행
    pre_std_filtered = []
    if pre_std_raw:
        # 사전규격은 품명(prdctClsfcNoNm) 외에 사업명(bsnsNm)으로도 검색
        pre_std_filtered = filter_data(pre_std_raw, prepared_detailed, ['prdctClsfcNoNm', 'bsnsNm'])
        log.append(f"사전규격 필터링 완료: {len(pre_std_filtered)}건.")


    all_found_data['pre_standard'] = pd.DataFrame(pre_std_filtered)
    
    pre_standard_map = {
        r['bfSpecRgstNo']: r for r in pre_std_filtered
        if r.get('bfSpecRgstNo')
    }

    # 3. 입찰공고
    log.append("\n========== 3. 입찰 공고 정보 수집 및 분석 ==========")
    bid_params = {'start_date': start_datetime_str, 'end_date': end_datetime_str}

    # 검색 필드: 입찰공고명(bidNtceNm) + 공고기관명(ntcelnsttNm)
    bid_df = collect_and_analyze(
        client.get_bid_announcements, bid_params, search_keywords, BROAD_KEYWORDS,
        search_fields=['bidNtceNm', 'ntcelnsttNm'],
        gemini_key=gemini_key, log_list=log, log_prefix="입찰공고", data_type="bid"
    )

    # 사전규격 연결
    if not bid_df.empty:
        def link_pre_standard(row):
            spec_no = row.get('bfSpecRgstNo')
            if spec_no and spec_no in pre_standard_map:
                return ("확인", spec_no, pre_standard_map[spec_no].get('registDt'))
            return ("해당 없음", None, None)
        bid_df[['prestandard_status', 'prestandard_no', 'prestandard_date']] = bid_df.apply(link_pre_standard, axis=1, result_type='expand')

    all_found_data['bid'] = bid_df
    # DB 저장
    if not bid_df.empty:
        upsert_project_data(bid_df, 'bid')

    # 4. 낙찰정보 (상태 업데이트용, 상세 키워드만 사용)
    log.append("\n========== 4. 낙찰 정보 수집 ==========")
    succ_bid_base_params = {'start_date': start_datetime_str, 'end_date': end_datetime_str}
    succ_dfs = []
    # prepared_detailed 키워드 재사용

    for code in ['1','2','3','5']:
        params_with_code = {**succ_bid_base_params, 'bsns_div_cd': code}

        # API 호출
        succ_raw = fetch_api_data(client.get_successful_bid_info, params_with_code, log, log_prefix=f"낙찰정보(코드:{code})")
        
        # succ_raw가 비어있지 않을 경우에만 필터링 수행
        succ_filtered = []
        if succ_raw:
            # 필터링 (상세 키워드, 다중 필드)
            succ_filtered = filter_data(succ_raw, prepared_detailed, ['bidNtceNm', 'ntcelnsttNm'])

        if succ_filtered:
            succ_dfs.append(pd.DataFrame(succ_filtered))
            log.append(f"낙찰정보(코드:{code}) 필터링 완료: {len(succ_filtered)}건.")

    all_found_data['successful_bid'] = pd.concat(succ_dfs, ignore_index=True) if succ_dfs else pd.DataFrame()
    if not all_found_data['successful_bid'].empty:
        upsert_project_data(all_found_data['successful_bid'], 'successful_bid')


    # 5. 계약정보 (상태 업데이트용, 상세 키워드만 사용)
    log.append("\n========== 5. 계약 정보 수집 ==========")
    contract_params = {'start_date': start_date_str, 'end_date': end_date_str}

    # API 호출
    contract_raw = fetch_api_data(client.get_contract_info, contract_params, log, log_prefix="계약정보")

    # contract_raw가 비어있지 않을 경우에만 필터링 수행
    contract_filtered = []
    if contract_raw:
        # 필터링 (상세 키워드, 다중 필드)
        contract_filtered = filter_data(contract_raw, prepared_detailed, ['cntrctNm', 'dminsttNm'])
        log.append(f"계약정보 필터링 완료: {len(contract_filtered)}건.")


    all_found_data['contract'] = pd.DataFrame(contract_filtered)
    
    if not all_found_data['contract'].empty:
        upsert_project_data(all_found_data['contract'], 'contract')


    # --- 보고서 생성 및 후처리 ---
    log.append("\n========== 6. 보고서 생성 및 전략 분석 시작 ==========")
    # 보고서 생성 (진단 로그가 포함된 함수 호출)
    report_dfs = create_report_data("procurement_data.db", log, min_relevance_score)

    risk_df, report_data_bytes, gemini_report = pd.DataFrame(), None, None

    if report_dfs:
        # 리스크 분석
        if "flat" in report_dfs and report_dfs["flat"] is not None:
             risk_df = analyze_project_risk(report_dfs["flat"])

        # 엑셀 시트 구성
        excel_sheets = {
            "종합 현황 보고서": report_dfs.get("structured"),
            "발주계획 현황": report_dfs.get("order_plan"),
            "리스크 분석": risk_df,
            # 원본 데이터는 수집된 전체 데이터 제공 (참고용)
            "발주계획 원본(수집 전체)": all_found_data.get('order_plan'),
            "입찰공고 원본(수집 전체)": all_found_data.get('bid'),
        }
        # 엑셀 파일 생성
        try:
            report_data_bytes = save_integrated_excel(excel_sheets)
            log.append("✅ 통합 엑셀 보고서 생성 완료.")
        except Exception as e:
            log.append(f"⚠️ 엑셀 파일 생성 중 오류 발생: {e}")

    # AI 전략 분석 및 키워드 확장 (API 키 유무 확인은 각 함수 내부에서 처리)
    if report_dfs and "flat" in report_dfs and report_dfs["flat"] is not None:
            # AI 전략 분석 (보고서 데이터 기준)
        gemini_report = get_gemini_analysis(gemini_key, report_dfs["flat"], log)

    # 키워드 확장 (수집된 모든 데이터를 통합하여 수행)
    if auto_expand_keywords:
        combined_df_list = [df for df in all_found_data.values() if df is not None and not df.empty]
        if combined_df_list:
            combined_df = pd.concat(combined_df_list, ignore_index=True)
            new_keywords = expand_keywords_with_gemini(gemini_key, combined_df, search_keywords, log)
            if new_keywords:
                updated_keywords = search_keywords.union(new_keywords)
                save_keywords(updated_keywords)
                log.append("🎉 키워드 파일이 새롭게 확장되었습니다!")


    # 최종 결과 반환
    return {
        "log": log,
        "risk_df": risk_df,
        "report_file_data": report_data_bytes,
        "report_filename": f"integrated_report_AI_{execution_time.strftime('%Y%m%d_%H%M%S')}.xlsx",
        "gemini_report": gemini_report
    }