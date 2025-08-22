import os
import requests
import ssl
import requests.adapters
import urllib3
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
# [신규 추가] 데이터 정합성 보장을 위한 UUID 임포트
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# --- 0. 회사 프로필 및 분석 설정 ---

# [강화됨] 회사 프로필 강화: 핵심 기술과 사업 분야의 교집합을 강조합니다.
COMPANY_PROFILE = """
우리 회사는 가상현실(VR/AR/XR/MR) 및 고정밀 시뮬레이션 기술을 기반으로 하는 [군사 및 경찰 훈련체계 전문 기업]입니다.

[핵심 역량 및 기술]
1. 위치 및 동작 인식 기술 기반의 정밀한 공간 정합 (현실 공간과 가상 환경의 실시간 매칭)
2. 몰입형 가상환경(디지털트윈) 구현 및 실시간 상호작용 (Haptics 포함)
3. 실제 장비(화기, 차량, 항공기 등) 연동 및 물리 엔진 기반 시뮬레이션 기술

[주요 사업 분야 (반드시 아래 항목에 해당해야 함)]
- 영상 모의 사격 훈련 시스템 및 과학화 사격장 구축 (군, 경찰, 예비군 포함)
- 가상 공수 강하(낙하산) 훈련 시뮬레이터
- 박격포/전차/항공기/함정 등 군사 장비 운용 및 전술 훈련 시뮬레이터
- 소부대 전투(CQB), 대테러, 재난 대응 훈련 시스템 (경찰특공대, 특전사 등)
- 예비군 과학화 훈련체계 구축 및 유지보수
- 군 장비 유지보수(MRO)를 위한 가상/증강현실 솔루션

[매우 중요: 관련성 판단 기준]
* 관련성이 높으려면 [기술 키워드(VR/시뮬레이션)]와 [고객/목적 키워드(군/경/훈련)]가 반드시 **동시에** 충족되어야 합니다.
* 예: '군사용 서버 구축' (X, 기술 불일치), '초등학교 VR 교실' (X, 고객/목적 불일치), '군사용 사격 시뮬레이터' (O)

[절대 아님 (관련성 0점 처리)]
- IT 인프라 (서버 가상화(Virtualization), VDI, 클라우드, 네트워크, 보안, 정보보호, 서버/PC 구매)
- 일반적인 시스템 통합(SI), 정보시스템(ERP, 그룹웨어), 웹사이트/앱 개발, 소프트웨어 라이선스 구매
- 단순 교육 및 연구 용역 (예: 리더십 교육, 직무 교육), 학술 연구
- 관련 없는 분야의 훈련 (스포츠/레저, 직업 훈련, 의료/간호 실습, 소방(구조/진압) 훈련)
- 건설, 토목, 시설 관리, 일반 장비 제조/구매 (예: 에어컨, 공조기, 생산 자동화, 로봇팔, 3D 프린터, 측정 장비)
"""

# 광범위 탐색용 키워드
BROAD_KEYWORDS = {"훈련", "체계", "시스템", "모의", "가상", "증강", "시뮬레이터", "시뮬레이션", "과학화", "교육", "연구개발", "성능개량", "군사", "경찰", "국방"}

# [개선됨] 네거티브 키워드 확장 및 체계화: PDF 분석 기반 오탐 키워드 대량 추가
# 이 목록에 포함된 단어가 사업명에 있으면 즉시 제외됩니다.
NEGATIVE_KEYWORDS = {
    # 가상화 및 IT 인프라 (Virtualization & IT Infra)
    "가상입찰", "가상계좌", "데스크톱가상화", "데스크탑가상화", "서버가상화", "VDI", "클라우드", "네트워크", "방화벽", "정보보호", "정보시스템", "ERP", "그룹웨어", "홈페이지",
    # 모의 관련 오탐 (False positives for 'Mock')
    "모의고사", "모의평가", "모의투자", "모의해킹",
    # 시스템/제어 관련 오탐 (산업/시설 제어 - PDF 기반 강화)
    "자동제어", "원격제어", "감시제어", "계장제어", "계측제어",
    "공조", "냉난방", "펌프", "배수", "정수", "하수", "오폐수", "맨홀", "VAV",
    "시스템에어컨", "냉난방시스템", "태양광시스템", "방송시스템", "음향시스템",
    # 관련 없는 분야 (Irrelevant Fields)
    "간호실습", "의료실습", "의료장비", "직업훈련", "일학습병행",
    "급식", "식재료", "토목", "건축", "시설관리", "체육용품",
    # 농업/기타 (PDF 기반 강화)
    "농업기계", "트랙터", "작물", "곡물", "버섯",
    # 일반 구매/라이선스
    "라이선스", "라이센스", "인쇄"
}


# --- 1. 보안 및 네트워크 설정 클래스 ---

# (기존 코드 유지)
class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, ssl_context=None, retry_strategy=None, **kwargs):
        self.ssl_context = ssl_context
        if retry_strategy:
            kwargs['max_retries'] = retry_strategy
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context
        )


# --- 1. 키워드 관리 (티어 시스템 도입) ---
# (기존 코드 유지)
KEYWORD_FILE = "keywords.txt"

# 1. 핵심 키워드 (CORE_KEYWORDS) - 100점 확정 키워드 (변경 없음)
CORE_KEYWORDS = {
    "영상사격", "모의사격", "영상 모의", "과학화 사격장",
    "공수강하", "가상강하", "낙하산 시뮬레이터",
    "박격포 시뮬레이터", "전차 시뮬레이터",
    "소부대 전투", "CQB", "대테러 훈련",
    "경찰특공대 훈련", "특전사 훈련",
    "예비군 과학화"
}

# [핵심 개선] 2. 일반 키워드 (GENERAL_KEYWORDS) - AI 분석이 필요한 광범위/모호한 키워드
# 이 목록을 대폭 확장하여, 모호한 단어가 자동으로 100점 처리되는 것을 방지합니다.
GENERAL_KEYWORDS = {
    # 기술 키워드
    "시뮬레이터", "시뮬레이션", "가상현실", "증강현실", "혼합현실",
    "VR", "AR", "MR", "XR", "디지털트윈", "메타버스",
    # 사업 유형 키워드
    "MRO", "유지보수", "성능개량", "체계개발", "연구개발",
    # 일반 명사 (오탐지 방지용 - 핵심)
    "체계", "훈련", "시스템", "장비", "교육", "솔루션", "플랫폼",
    "제어", "통제", "감시", "AI", "인공지능", "로봇", "자동화", "측정",
    # 관련 가능성이 있는 분야 (단독으로는 모호함)
    "드론", "항공", "전차", "박격포", "지뢰", "함정", "화생방", "군사", "경찰", "국방"
}

INITIAL_KEYWORDS = list(CORE_KEYWORDS.union(GENERAL_KEYWORDS))

def load_keywords(initial_keywords: list) -> set:
    if not os.path.exists(KEYWORD_FILE):
        save_keywords(set(initial_keywords))
        return set(initial_keywords)
    try:
        with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
            loaded = {line.strip() for line in f if line.strip()}
            return loaded if loaded else set(initial_keywords)
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

# [중요] 100점 매칭 대상 키워드 선정 로직: (로드된 전체 키워드 - 일반 키워드) + 핵심 키워드
# 즉, GENERAL_KEYWORDS에 포함된 단어는 100점 매칭에서 제외됩니다.
def get_strict_match_keywords(loaded_keywords: set) -> set:
    return (loaded_keywords - GENERAL_KEYWORDS).union(CORE_KEYWORDS)


# --- 2. 유틸리티 및 API 클라이언트 ---

# (safe_int, format_price, save_integrated_excel 함수는 기존 코드 유지 - 변경 없음)
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
    """통합 엑셀 저장."""
    output = io.BytesIO()
    try:
        import xlsxwriter.utility

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- 1. 스타일 정의 ---
            order_plan_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#FFF2CC', 'border': 1})
            main_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#D3D3D3', 'border': 1})
            ai_analysis_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#E2F0D9', 'border': 1})
            
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
                    # MultiIndex 헤더 처리
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
                    start_row = 2
                else:
                    # 일반 헤더 처리
                    for col_num, value in enumerate(df_data.columns.values):
                        current_format = order_plan_header_format
                        if 'AI' in value: current_format = ai_analysis_header_format
                        worksheet.write(0, col_num, value, current_format)
                    start_row = 1

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
                        
                    is_reason_col = (is_multi_index and headers.get_level_values(1)[i] == '분석 이유') or \
                                    (not is_multi_index and headers[i] == 'AI 분석 이유')
                    
                    # [수정됨] 분석 이유 컬럼 너비 확장 (CoT 도입으로 내용이 길어짐)                
                    width = min(100, max(25, header_len, data_max_len) + 2) if is_reason_col else min(60, max(10, header_len, data_max_len) + 2)
                    
                    worksheet.set_column(i, i, width)
                    col_widths.append(width)

                # --- 4. 데이터 작성 및 행 높이 자동 조절 ---
                for row_num, row_data in enumerate(data_for_calc, start=start_row):
                    max_lines = 1
                    for i, cell_text in enumerate(row_data):
                        if len(str(cell_text)) > 0 and col_widths[i] > 0:
                            # 보정계수 1.8은 한글/영문 혼용 환경에서 평균적으로 잘 동작하는 값
                            try:
                                lines_needed = -(-len(str(cell_text)) // (col_widths[i] / 1.8))
                                if lines_needed > max_lines:
                                    max_lines = lines_needed
                            except ZeroDivisionError:
                                 pass
                    
                    row_height = 15 * max_lines
                    worksheet.set_row(row_num, min(400, row_height))
                    
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
    # (기존 코드 유지, 안정성 개선 포함)
    def __init__(self, service_key: str):
        if not service_key: raise ValueError("서비스 키는 필수입니다.")
        self.service_key = service_key
        self.base_url_std = "https://apis.data.go.kr/1230000/ao/PubDataOpnStdService"
        self.base_url_plan = "https://apis.data.go.kr/1230000/ao/OrderPlanSttusService"

        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        # SSL 설정은 환경에 따라 조정이 필요할 수 있습니다. 문제가 지속되면 아래 주석을 해제하세요.
        # ctx.set_ciphers('ALL:@SECLEVEL=1') 
        
        # 재시도 전략 강화
        retry_strategy = Retry(
            total=5, # 재시도 횟수 증가
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1.5 # 재시도 간격 증가
        )

        self.session = requests.Session()
        adapter = CustomHttpAdapter(ssl_context=ctx, retry_strategy=retry_strategy)
        self.session.mount("https://", adapter)

    # (기존 코드 유지 - _make_request 및 기타 API 호출 함수들, 변경 없음)
    def _make_request(self, base_url: str, endpoint: str, params: dict, log_list: list):
        
        decoded_key = "NOT_SET"
        response = None

        try:
            try:
                decoded_key = urllib.parse.unquote(self.service_key)
            except Exception as e:
                log_list.append(f"⚠️ [내부오류] ({type(e).__name__}) 서비스 키 디코딩 실패: {e}.")
                return []

            url = f"{base_url}/{endpoint}?ServiceKey={decoded_key}"

            other_params = {
                'pageNo': 1,
                'numOfRows': 999,
                'type': 'json',
                **params
            }
            
            # 타임아웃을 120초로 연장
            response = self.session.get(url, params=other_params, timeout=120)
            response.raise_for_status()

            content = response.text.strip()
            if not content:
                log_list.append(f"ℹ️ [API통신] 응답 본문이 비어 있습니다 ({endpoint}). Status: {response.status_code}.")
                return []

            data = response.json()

        except requests.exceptions.SSLError as e:
            log_list.append(f"⚠️ [SSL오류] ({type(e).__name__}) SSL 통신 오류 발생 ({endpoint}): {e}")
            return []
        
        except requests.exceptions.HTTPError as e:
            log_list.append(f"⚠️ [HTTP오류] ({type(e).__name__}) 서버 에러 발생 ({endpoint}): {e}. (자동 재시도 실패)")
            # ... (에러 로깅 코드 생략) ...
            return []

        except (requests.exceptions.JSONDecodeError, json.JSONDecodeError, ValueError) as e:
            log_list.append(f"⚠️ [JSON파싱오류] ({type(e).__name__}) API 응답 형식이 JSON이 아닙니다 ({endpoint}). 오류: {e}")
            # ... (에러 로깅 코드 생략) ...
            return []
        
        except requests.exceptions.RequestException as e:
            log_list.append(f"⚠️ [네트워크오류] ({type(e).__name__}) 네트워크 연결 실패 또는 타임아웃 ({endpoint}): {e}. (자동 재시도 실패)")
            return []
        
        except Exception as e:
            log_list.append(f"⚠️ [예상치못한오류] ({type(e).__name__}) 요청 처리 중 알 수 없는 오류 발생 ({endpoint}): {e}")
            return []


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

    def get_order_plans(self, year, log_list):
        # ... (함수 내용 생략) ...
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


# --- 3. 최적화된 데이터 처리 함수 (API 호출 단일화) ---
# (기존 코드 유지)

def prepare_keywords(keywords: Set[str]) -> List[Tuple[str, str]]:
    # ... (함수 내용 생략) ...
    prepared = []
    for kw in keywords:
        kw_lower = kw.lower()
        kw_no_space = kw_lower.replace(" ", "")
        if kw_no_space:
             prepared.append((kw_lower, kw_no_space))
    return prepared

def prepare_negative_keywords(keywords: Set[str]) -> List[str]:
    # ... (함수 내용 생략) ...
    prepared = []
    for kw in keywords:
        kw_no_space = kw.lower().replace(" ", "")
        if kw_no_space:
             prepared.append(kw_no_space)
    return prepared


def fetch_api_data(fetch_function, params, log_list, log_prefix="") -> List[dict]:
    # ... (함수 내용 생략) ...
    is_order_plan = log_prefix == "발주계획"

    if not is_order_plan:
        log_list.append(f"[{log_prefix}] API 데이터 요청 시작...")

    raw_data = fetch_function(log_list=log_list, **params)

    if not is_order_plan:
        if not raw_data:
            is_handled = False
            if log_list:
                 is_handled = log_list[-1].startswith("⚠️") or log_list[-1].startswith("ℹ️") or log_list[-1].startswith("💡")
            if not is_handled:
                log_list.append("API에서 수신된 데이터가 없습니다.")
            return []
        log_list.append(f"API로부터 총 {len(raw_data)}건 수신.")

    return raw_data if raw_data else []

# [중요] 필터링 함수: 네거티브 키워드가 적용되는 핵심 로직
def filter_data(data: List[dict], prepared_keywords: List[Tuple[str, str]], search_fields: List[str], prepared_negative_keywords: Optional[List[str]] = None) -> List[dict]:
    # ... (함수 내용 생략) ...
    filtered_data = []

    for item in data:
        if not isinstance(item, dict): continue

        match_found = False
        is_negative = False
        
        for field in search_fields:
            field_value = item.get(field)
            if not field_value: continue

            target_text = str(field_value).lower()
            target_text_no_space = target_text.replace(" ", "")

            # [핵심] 네거티브 키워드 검사
            if prepared_negative_keywords:
                for neg_kw in prepared_negative_keywords:
                    if neg_kw in target_text_no_space:
                        is_negative = True
                        break
                if is_negative:
                    break # 네거티브 키워드가 발견되면 다른 필드 검사 중단

            # 키워드 매칭 검사 (네거티브가 아닐 경우에만 수행됨)
            if not is_negative:
                for kw_lower, kw_no_space in prepared_keywords:
                    if kw_lower in target_text:
                        match_found = True
                        break
                    if kw_no_space in target_text_no_space:
                         match_found = True
                         break
                if match_found:
                    break

        # 최종 판단: 키워드가 매칭되었고 AND 네거티브 키워드에 해당하지 않아야 함
        if match_found and not is_negative:
            filtered_data.append(item)

    return filtered_data

# --- 4. 데이터베이스 ---
# (기존 코드 유지, 변경 없음)

def setup_database():
    # ... (함수 내용 생략) ...
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS projects (bidNtceNo TEXT PRIMARY KEY, bidNtceNm TEXT, ntcelnsttNm TEXT, presmptPrce INTEGER, bid_status TEXT DEFAULT '공고', bidNtceDate TEXT, sucsfCorpNm TEXT, cntrctAmt INTEGER, cntrctDate TEXT)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_year TEXT, category TEXT, dminsttNm TEXT, prdctNm TEXT, asignBdgtAmt INTEGER,
            orderInsttNm TEXT, orderPlanPrd TEXT, cntrctMthdNm TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderPlanPrd)
        )
    """)

    def add_column_if_not_exists(table, column, definition):
        try:
            cols = [info[1] for info in cursor.execute(f"PRAGMA table_info({table})")]
            if column not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception as e:
            logging.warning(f"Error altering table {table}: {e}")

    add_column_if_not_exists("projects", "prestandard_status", "TEXT")
    add_column_if_not_exists("projects", "prestandard_no", "TEXT")
    add_column_if_not_exists("projects", "prestandard_date", "TEXT")

    for table in ["projects", "order_plans"]:
        add_column_if_not_exists(table, "relevance_score", "INTEGER")
        add_column_if_not_exists(table, "relevance_reason", "TEXT")
        add_column_if_not_exists(table, "collection_method", "TEXT")

    conn.commit(); conn.close()

def upsert_project_data(df, stage):
    # ... (함수 내용 생략) ...
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()

    for _, r in df.iterrows():
        try:
            if stage == 'bid':
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
    # ... (함수 내용 생략) ...
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()

    for _, r in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO order_plans
                (plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderInsttNm, orderPlanPrd, cntrctMthdNm, relevance_score, relevance_reason, collection_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderPlanPrd) DO UPDATE SET
                    relevance_score=excluded.relevance_score,
                    relevance_reason=excluded.relevance_reason,
                    collection_method=excluded.collection_method,
                    created_at=CURRENT_TIMESTAMP
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

# [핵심 수정] AI 관련성 점수 계산 함수 - UUID 기반 정합성 보장 및 고도화된 CoT 프롬프트 적용
def calculate_ai_relevance(api_key, df_input, data_type, log_list):
    """Gemini를 사용하여 관련성 점수를 계산합니다 (UUID 기반 데이터 정합성 보장 및 CoT 적용)."""
    
    # (이하 AI 분석 로직은 이전 답변과 동일하게 유지 - 변경 없음)
    if df_input.empty:
        return df_input.copy()
    
    df = df_input.copy()

    # AI 처리를 위한 임시 고유 ID 생성 (UUID 사용)
    AI_TEMP_ID_COL = '_ai_temp_merge_id'
    df[AI_TEMP_ID_COL] = [str(uuid.uuid4()) for _ in range(len(df))]

    # 결과 컬럼 초기화
    df['relevance_score'] = -1 # 분석 오류나 처리 실패 시 -1로 표시
    df['relevance_reason'] = "분석 대기 또는 실패"

    if not api_key:
         log_list.append("ℹ️ Gemini API 키가 없어 AI 관련성 분석을 생략합니다.")
         df['relevance_score'] = pd.NA
         df['relevance_reason'] = 'API 키 없음'
         # 임시 컬럼 삭제 후 반환
         return df.drop(columns=[AI_TEMP_ID_COL], errors='ignore')

    fields_map = {
        'order_plan': {'title': 'prdctNm', 'org': 'dminsttNm', 'budget': 'asignBdgtAmt', 'category': 'category'},
        'bid': {'title': 'bidNtceNm', 'org': 'ntcelnsttNm', 'budget': 'presmptPrce'}
    }

    if data_type not in fields_map:
        return df.drop(columns=[AI_TEMP_ID_COL], errors='ignore')

    fields = fields_map[data_type]
    log_list.append(f"🤖 AI 관련성 분석 시작 ({data_type}, 총 {len(df)}건)...")

    # [핵심 수정] 프롬프트 정의 - CoT 강화, 평가 기준 구체화 및 교집합 요구
    PROMPT_TEMPLATE = (
        "당신은 군사/경찰 훈련 시뮬레이션(VR/XR) 전문 기업의 조달 분석 AI입니다. 아래 [회사 프로필]을 기준으로, 제공된 [사업 목록]의 관련성을 '매우 엄격하고 비판적으로' 평가하세요.\n\n"
        f"[회사 프로필]\n{COMPANY_PROFILE}\n\n"
        "----------------------\n"
        "[평가 절차 (Chain-of-Thought)]\n"
        "각 사업에 대해 다음 3단계를 거쳐 최종 점수를 결정하고, 이 과정을 'reason' 필드에 간결하게 요약하세요.\n"
        "1단계 (요소 분석): 사업명(title)과 기관명(organization)에서 [기술 요소(VR/시뮬레이션 등)]와 [목적/고객 요소(군/경/훈련 등)]를 각각 식별합니다.\n"
        "2단계 (교집합 검증): 식별된 두 요소가 회사의 핵심 사업 분야와 교집합을 이루는지 판단합니다. (가장 중요: 둘 다 충족해야 함)\n"
        "3단계 (점수 결정): [평가 기준]에 따라 점수를 부여하고 이유를 설명합니다.\n"
        "----------------------\n"
        "[매우 중요 지침]\n"
        "1. 교집합 원칙: '군사용'이라도 IT 인프라나 단순 장비 구매는 0점입니다. 'VR/시뮬레이션'이라도 군/경 훈련 목적이 아니면 0점입니다.\n"
        "2. 모호성 금지: '시스템', '체계', '성능개량', '장비' 같은 단어만으로는 판단하지 마세요. 구체적인 내용을 확인해야 합니다. 모호하면 낮은 점수(0~30점)를 부여하세요.\n"
        "3. 용어 구분: '가상화(Virtualization/VDI)'는 '가상현실(VR)'이 아닙니다. '자동제어/감시제어'는 훈련 시뮬레이션이 아닙니다.\n"
        "4. 환각 금지: 평가는 제공된 정보에만 기반해야 합니다. 내용을 추측하지 마세요.\n"
        "5. 데이터 정합성: 입력된 고유 ID(id)는 결과 JSON의 'id' 필드에 그대로 반환해야 합니다.\n"
        "----------------------\n"
        "[평가 기준 (100점 만점) - 매우 엄격 적용]\n"
        "- 90~100점: 핵심 주력 분야(사격, CQB, 공수강하, 전차/박격포 시뮬레이터)와 정확히 일치하며 VR/XR/시뮬레이션 기술이 명확한 경우.\n"
        "- 70~89점: 군/경 대상의 시뮬레이터/가상현실 훈련 사업이며 관련성이 매우 높은 경우 (예: 과학화 훈련장, MRO 솔루션).\n"
        "- 50~69점 (잠재적 관심): 군/경 관련 사업이지만 시뮬레이션/VR 연관성이 불분명하거나 낮지만 가능성이 있는 경우.\n"
        "- 31~49점 (가능성 낮음): 관련성은 낮으나 특정 기술 요소가 포함되어 검토 여지가 있는 경우.\n"
        "- 0~30점 (관련 없음): [절대 아님]에 해당하거나, 기술/목적 교집합이 없는 경우 (예: 일반 장비 구매, 시설 제어 시스템).\n\n"
        "[출력 형식]\n"
        "결과는 반드시 아래 형식의 JSON 리스트로만 출력해야 합니다.\n"
        "예시: [{\"id\": \"uuid-1\", \"score\": 85, \"reason\": \"1)요소: 항공 시뮬레이터(기술), 국방부(고객). 2)검증: 군 항공 훈련 시뮬레이터로 교집합 확인. 3)결정: 관련성 높음.\"}, "
        "{\"id\": \"uuid-2\", \"score\": 5, \"reason\": \"1)요소: 서버 가상화(기술), 국방부(고객). 2)검증: 기술(가상화)이 회사 분야 아님. 3)결정: [절대 아님]에 해당.\"}]\n\n"
        "[사업 목록]\n"
        "{data_placeholder}"
    )

    try:
        genai.configure(api_key=api_key)
        # 정확도 향상을 위해 Gemini Pro 모델 사용.
        model_name = 'gemini-1.5-pro-latest' 
        # model_name = 'gemini-1.5-flash' # 속도가 매우 중요할 경우 Flash 사용
        model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})

        results = []
        # Pro 모델은 Rate Limit을 고려하여 배치 크기 및 지연 시간 설정
        BATCH_SIZE = 25 
        SLEEP_TIME = 2.5

        # (AI 호출 및 결과 병합 로직은 기존 코드 유지 - UUID 기반 안정적 병합)
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i+BATCH_SIZE]
            data_list = []

            # 배치 데이터를 JSON 리스트 형식으로 구성 (고유 ID 포함)
            for index, row in batch_df.iterrows():
                item = {
                    "id": row[AI_TEMP_ID_COL], # [수정] 임시 고유 ID 사용
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
                
                if not response.text or response.text.strip() == "":
                    raise ValueError("AI 응답이 비어 있음")

                batch_results = json.loads(response.text)

                if isinstance(batch_results, list):
                    # [수정] 결과에 id가 포함되어 있는지 검증
                    valid_batch_results = [res for res in batch_results if isinstance(res, dict) and 'id' in res and res['id'] is not None]
                    
                    # [강화된 검증] AI가 요청한 ID를 모두 반환했는지 확인 (데이터 누락/환각 감지)
                    requested_ids = set(batch_df[AI_TEMP_ID_COL])
                    returned_ids = {res['id'] for res in valid_batch_results}
                    
                    if requested_ids != returned_ids:
                        missing_count = len(requested_ids - returned_ids)
                        extra_count = len(returned_ids - requested_ids)
                        log_list.append(f"⚠️ 데이터 불일치 경고 (배치 {i}~{i+BATCH_SIZE}): 요청/응답 ID 불일치. 누락: {missing_count}, 추가(환각): {extra_count}. 유효한 데이터만 처리됩니다.")

                    results.extend(valid_batch_results)
                else:
                    raise ValueError("응답이 JSON 리스트 형식이 아님")

                log_list.append(f"  - AI 분석 진행 중 ({min(i+BATCH_SIZE, len(df))}/{len(df)})... (Model: {model_name})")
                time.sleep(SLEEP_TIME)

            except (json.JSONDecodeError, ValueError, Exception) as e:
                log_list.append(f"⚠️ AI 응답 처리 오류 (배치 {i}~{i+BATCH_SIZE}): {e}. 해당 배치는 생략합니다. (모델: {model_name})")


        # --- [핵심 수정: 결과 병합 로직 (UUID 기반 df.update)] ---
        if results:
            # 원본 인덱스 보존
            original_index = df.index
            
            try:
                results_df = pd.DataFrame(results)

                if 'id' not in results_df.columns:
                     log_list.append(f"⚠️ 치명적 오류: AI 응답에 필수 'id' 필드가 없습니다. 병합을 중단합니다.")
                     return df.drop(columns=[AI_TEMP_ID_COL], errors='ignore')

                # 'id'(UUID)를 인덱스로 설정
                results_df = results_df.set_index('id')

                # 점수 데이터 타입 변환 및 정리
                results_df['score'] = pd.to_numeric(results_df['score'], errors='coerce').fillna(-1)
                results_df['score'] = results_df['score'].apply(lambda x: max(0, min(100, int(x))) if x >= 0 else -1) # 점수 범위 0-100으로 제한

                # 업데이트할 컬럼 정의
                update_cols = {'score': 'relevance_score'}
                if 'reason' in results_df.columns:
                    update_cols['reason'] = 'relevance_reason'

                update_data = results_df[list(update_cols.keys())].rename(columns=update_cols)

                # 원본 데이터프레임(df)도 임시 ID를 인덱스로 설정하여 업데이트 수행
                df = df.set_index(AI_TEMP_ID_COL)
                
                # df.update() 실행: 고유 ID(UUID)를 기준으로 안정적으로 값을 덮어씀
                df.update(update_data)
                
                # 원본 인덱스로 복원
                df.index = original_index

                # 진단 로그 추가
                success_count = (df['relevance_score'] >= 0).sum()
                error_count = (df['relevance_score'] == -1).sum()
                log_list.append(f"📊 AI 분석 결과 매칭 완료: 성공 {success_count}건 / 실패(오류 또는 미처리) {error_count}건.")

            except Exception as e:
                log_list.append(f"⚠️ AI 결과 병합 중 치명적 오류 발생: {e}. 데이터가 섞였을 수 있습니다.")
                logging.exception(e)
                # 오류 발생 시 인덱스 복원 시도
                if hasattr(df, 'index') and df.index.name == AI_TEMP_ID_COL:
                     try:
                         df.index = original_index
                     except Exception:
                         pass

        # 임시 컬럼 최종 제거
        df = df.drop(columns=[AI_TEMP_ID_COL], errors='ignore')

        log_list.append("✅ AI 관련성 분석 완료.")
        try:
            df['relevance_score'] = pd.to_numeric(df['relevance_score'], errors='coerce').fillna(-1).astype(int)
        except Exception:
             pass
        return df

    except Exception as e:
        log_list.append(f"⚠️ AI 관련성 분석 중 치명적 오류 발생: {e}")
        logging.exception(e)
        # 오류 발생 시에도 임시 컬럼 삭제 후 반환
        return df.drop(columns=[AI_TEMP_ID_COL], errors='ignore')

# [개선됨] 전략 분석 함수 (기존 코드 유지, 변경 없음)
def get_gemini_analysis(api_key, df, log_list):
    if df.empty: log_list.append("AI가 분석할 데이터가 없습니다."); return None

    # 관련성 점수가 높은 상위 데이터만 필터링하여 분석 효율성 및 품질 향상 (50점 기준)
    df_high_relevance = df[df['relevance_score'] >= 50].copy()
    if df_high_relevance.empty:
         log_list.append("AI 전략 분석 대상 데이터(50점 이상)가 없습니다."); return None

    if not api_key:
        log_list.append("ℹ️ Gemini API 키가 없어 전략 분석을 생략합니다.")
        return None

    try:
        # 전략 분석은 정확도를 위해 Pro 모델 사용
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-pro-latest')
        log_list.append("Gemini API로 맞춤형 전략 분석 시작 (Model: Pro)...")

        # ... (나머지 함수 내용 생략 - 기존과 동일) ...
        df_for_prompt = df_high_relevance.copy()
        if 'presmptPrce' in df_for_prompt.columns:
             df_for_prompt['presmptPrce'] = df_for_prompt['presmptPrce'].apply(format_price)
        if 'cntrctAmt' in df_for_prompt.columns:
             df_for_prompt['cntrctAmt'] = df_for_prompt['cntrctAmt'].apply(format_price)

        # 분석 대상 데이터 수를 50개로 제한하여 프롬프트 길이 최적화
        data_for_prompt_str = df_for_prompt[[c for c in ['relevance_score', 'bidNtceNm', 'bid_status', 'ntcelnsttNm', 'presmptPrce', 'cntrctAmt', 'relevance_reason'] if c in df_for_prompt.columns]].head(50).to_string()

        # [개선됨] 전략 분석 프롬프트 수정: 핵심 기술 연계 강조 및 구체적인 제언 요구
        prompt = (
            f"당신은 아래 [회사 프로필]을 보유한 기업의 '신사업 전략팀장'입니다. 제공된 나라장터 조달 데이터(관련성 점수 50점 이상)를 바탕으로 심층 분석 보고서를 작성해주세요. 데이터의 'relevance_score'가 높은 사업에 집중하여 분석해주세요.\n\n"
            f"[회사 프로필]\n{COMPANY_PROFILE}\n\n"
            f"[분석 데이터]\n{data_for_prompt_str}\n\n---\n"
            "## 군·경 훈련 시뮬레이션(VR/XR) 사업 기회 심층 분석 보고서\n\n"
            "### 1. 총평 및 시장 동향\n"
            "(우리 회사의 핵심 기술(공간 정합, VR/XR 시뮬레이션)과 관련성이 높은 군/경 훈련 시장의 최신 동향과 특징을 분석해주세요. 특히 어떤 종류의 시뮬레이션 수요가 증가하고 있는지 언급해주세요.)\n\n"
            "### 2. 주요 사업 심층 분석 (관련성 점수 상위 3~5개)\n"
            "(관련성 점수(relevance_score)가 가장 높은 프로젝트를 선정하여 아래 표 형식으로 분석해주세요. '분석 및 제언' 항목에는 **우리 회사의 핵심 기술(예: 공간 정합, 위치/동작 인식, 실제 장비 연동)을 어떻게 차별화 포인트로 활용할 수 있을지** 구체적으로 작성해주세요.)\n\n"
            "| 사업명 | 발주기관 | 관련성 점수 | 추정가격/계약금액 | 분석 및 제언 (자사 기술 연계 및 차별화 방안) |\n"
            "|---|---|---|---|---|\n"
            "| (사업명) | (기관명) | (점수) | (금액) | (예: 이 CQB 훈련 사업은 **현실 공간과 가상 시나리오를 정밀하게 매칭하는 우리 기술**이 핵심 경쟁력임. 실제 화기 연동 기술을 제안하여 차별화 가능.) |\n\n"
            "### 3. 기술 확장 기회 및 키워드\n"
            "(데이터에서 식별된 내용 중, 우리 회사가 새롭게 주목해야 할 기술 트렌드나 확장 가능한 사업 영역을 1~2가지 제시하고, 관련 핵심 키워드를 5개 이상 나열해주세요.)\n\n"
            "### 4. 차기 사업 전략 제언\n"
            "(위 분석을 종합하여, 우리 회사가 향후 6개월간 집중해야 할 구체적인 실행 전략(예: 특정 기관 타겟팅, 특정 기술 고도화, 파트너십 구축 등)을 2가지 제언해주세요.)"
        )
        response = model.generate_content(prompt); log_list.append("Gemini 맞춤형 전략 분석 완료.")
        return response.text
    except Exception as e:
        log_list.append(f"⚠️ Gemini API 호출 중 오류 발생: {e}")
        return None

# [개선됨] 키워드 확장 함수 (기존 코드 유지, 변경 없음)
def expand_keywords_with_gemini(api_key, df, existing_keywords, log_list):
    if df.empty: log_list.append("키워드 확장을 위한 분석 데이터가 없습니다."); return set()

    # 관련성 점수가 높은 데이터에서만 키워드 추출하여 품질 향상 (70점 기준)
    df_high_relevance = df[df.get('relevance_score', 0) >= 70].copy()
    if df_high_relevance.empty:
         log_list.append("키워드 확장을 위한 고품질 데이터(70점 이상)가 없습니다."); return set()

    if not api_key:
        log_list.append("ℹ️ Gemini API 키가 없어 키워드 확장을 생략합니다.")
        return set()

    try:
        # 키워드 확장도 정확도를 위해 Pro 모델 사용
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-pro-latest')
        log_list.append("Gemini API로 지능형 키워드 확장 시작 (Model: Pro)...")
        # ... (나머지 함수 내용 생략 - 기존과 동일) ...
        project_titles = pd.concat([df_high_relevance[col] for col in ['bidNtceNm', 'cntrctNm', 'prdctClsfcNoNm', 'prdctNm'] if col in df_high_relevance.columns]).dropna().unique()
        project_titles_str = '\n- '.join(project_titles[:50]) # 최대 50개로 제한

        # [개선됨] 키워드 확장 프롬프트 수정: 회사 프로필 제공 및 일반 명사 제외 요청
        prompt = (
            f"당신은 아래 [회사 프로필]을 가진 기업의 조달 정보 분석가입니다. "
            "아래는 현재 우리가 사용중인 검색 키워드와, 최근 조달 시스템에서 발견된 관련성 높은 사업명/품명 목록입니다. "
            "이 정보를 바탕으로, 우리 회사의 사업과 관련성이 높으면서도 기존 키워드에 없는 **새로운 검색 키워드를 5~10개 추천**해주세요. "
            "**[중요] '시스템', '장비', '개발'과 같은 일반 명사는 제외하고, 구체적인 기술이나 사업 분야를 나타내는 키워드만 추천해야 합니다.**\n\n"
            f"[회사 프로필]\n{COMPANY_PROFILE}\n\n"
            f"[기존 키워드]\n{', '.join(sorted(list(existing_keywords)))}\n\n"
            f"[최근 발견된 관련성 높은 사업명/품명]\n- {project_titles_str}\n\n"
            "[추천 키워드 (쉼표로 구분된 목록만 제공)]"
        )
        response = model.generate_content(prompt)
        new_keywords = {k.strip() for k in response.text.strip().split(',') if k.strip() and len(k.strip()) > 1}
        
        # [신규] 새로 추천된 키워드가 GENERAL_KEYWORDS에 포함되는지 확인하여 로그 기록
        generic_filtered_out = new_keywords.intersection(GENERAL_KEYWORDS)
        if generic_filtered_out:
             log_list.append(f"ℹ️ Gemini 추천 키워드 중 일반 키워드(자동 100점 방지)로 분류됨: {generic_filtered_out}")

        log_list.append(f"🎉 Gemini가 추천한 신규 키워드: {new_keywords}")
        return new_keywords
    except Exception as e:
        log_list.append(f"⚠️ Gemini 키워드 확장 중 오류 발생: {e}")
        return set()

def analyze_project_risk(df: pd.DataFrame) -> pd.DataFrame:
    # (기존 코드 유지, 기준 일부 조정)
    # ... (함수 내용 생략) ...
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
            if days_elapsed > 45: # 기준 강화 (30일 -> 45일)
                score += 5
                reasons.append(f'공고 후 {days_elapsed}일 경과 (유찰 가능성)')

        if row.get('prestandard_status') == '해당 없음':
            score += 3
            reasons.append('사전규격 미공개')

        price = row.get('presmptPrce')
        if pd.notna(price) and isinstance(price, (int, float)) and 0 < price < 100000000: # 기준 변경 (5천만 -> 1억)
            score += 2
            reasons.append('소규모 사업 (1억원 미만)')

        ongoing_df.loc[index, 'score'] = score
        ongoing_df.loc[index, 'risk_reason'] = ', '.join(reasons)

    ongoing_df['risk_level'] = ongoing_df['score'].apply(lambda s: '높음' if s >= 7 else ('보통' if s >= 4 else '낮음'))

    risk_table = ongoing_df[['bidNtceNm', 'ntcelnsttNm', 'bid_status', 'risk_level', 'risk_reason']]
    return risk_table.rename(columns={'bidNtceNm':'사업명','ntcelnsttNm':'발주기관','bid_status':'진행 상태','risk_level':'리스크 등급','risk_reason':'주요 리스크'}).sort_values(by='리스크 등급',key=lambda x:x.map({'높음':0,'보통':1,'낮음':2}))


# 보고서 생성 함수
def create_report_data(db_path, log_list, min_relevance_score=0):
    # (기존 코드 유지, 변경 없음)
    # ... (함수 내용 생략) ...
    log_list.append(f"DB에서 최종 데이터 조회 중 (최소 관련성 점수: {min_relevance_score})...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    report_data = {}

    try:
        # 1. 프로젝트(입찰공고 이후) 데이터 처리
        try:
            total_count_query = "SELECT COUNT(*) FROM projects WHERE relevance_score IS NOT NULL AND relevance_score >= 0"
            cursor.execute(total_count_query)
            result = cursor.fetchone()
            total_projects_scored = result[0] if result else 0

            query = f"""
                SELECT * FROM projects
                WHERE relevance_score >= ?
                ORDER BY relevance_score DESC, bidNtceDate DESC
            """
            all_projects_df = pd.read_sql_query(query, conn, params=(min_relevance_score,))

            log_list.append(f"📊 DB 프로젝트 현황: 총 {total_projects_scored}건 평가됨 -> {len(all_projects_df)}건이 임계값({min_relevance_score}점) 통과.")

        except pd.errors.DatabaseError as e:
            log_list.append(f"⚠️ 프로젝트 DB 조회 중 오류: {e}")
            all_projects_df = pd.DataFrame()


        if not all_projects_df.empty:
            flat_df = all_projects_df.copy()

            report_data["flat"] = flat_df.copy()

            for col in ['prestandard_date','bidNtceDate','cntrctDate']:
                if col in flat_df.columns:
                    flat_df[col] = pd.to_datetime(flat_df[col],errors='coerce').dt.strftime('%Y-%m-%d')

            for col in ['presmptPrce','cntrctAmt']:
                if col in flat_df.columns:
                    flat_df[col] = flat_df[col].apply(format_price)

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

            query_plan = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER(PARTITION BY plan_year, category, dminsttNm, prdctNm ORDER BY created_at DESC) as rn
                    FROM order_plans
                    WHERE relevance_score >= ?
                ) WHERE rn = 1
                ORDER BY relevance_score DESC, asignBdgtAmt DESC
            """
            all_order_plans_df = pd.read_sql_query(query_plan, conn, params=(min_relevance_score,))

            log_list.append(f"📊 DB 발주계획 현황: 총 {total_plans_scored}건 평가됨 -> {len(all_order_plans_df)}건이 임계값({min_relevance_score}점) 통과.")

        except pd.errors.DatabaseError as e:
             log_list.append(f"⚠️ 발주계획 DB 조회 중 오류: {e}")
             all_order_plans_df = pd.DataFrame()

        if not all_order_plans_df.empty:
            order_plan_df = all_order_plans_df.copy()

            order_plan_df['asignBdgtAmt_formatted'] = order_plan_df['asignBdgtAmt'].apply(format_price)

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

# (기존 코드 유지, 변경 없음)
def collect_and_analyze(fetch_function, params, detailed_keywords: Set[str], broad_keywords: Set[str], search_fields: Union[str, List[str]], gemini_key, log_list, log_prefix, data_type, negative_keywords: Set[str]):
    # ... (함수 내용 생략) ...
    if isinstance(search_fields, str):
        search_fields_list = [search_fields]
    else:
        search_fields_list = search_fields

    log_list.append(f"\n--- [{log_prefix}] 1단계: 통합 데이터 수집 시작 ---")

    raw_data = fetch_api_data(fetch_function, params, log_list, log_prefix)

    if not raw_data:
        return pd.DataFrame()

    prepared_neg_keywords = prepare_negative_keywords(negative_keywords)

    log_list.append(f"광범위 키워드 필터링 시작 (네거티브 키워드 적용)...")
    prepared_broad_keywords = prepare_keywords(broad_keywords)
    
    # 1단계 필터링: 광범위 키워드 사용 + 네거티브 키워드 적용
    broad_data = filter_data(raw_data, prepared_broad_keywords, search_fields_list, prepared_neg_keywords)
    log_list.append(f"광범위 키워드 필터링 후 {len(broad_data)}건 확보.")

    if not broad_data:
        return pd.DataFrame()

    log_list.append(f"\n--- [{log_prefix}] 2단계: 상세(엄격한) 키워드 매칭 (Score 100) ---")

    prepared_detailed_keywords = prepare_keywords(detailed_keywords)

    # 2단계 필터링: 상세(엄격한) 키워드 사용 + 네거티브 키워드 적용 (혹시 모를 경우 대비)
    keyword_hits_data = filter_data(broad_data, prepared_detailed_keywords, search_fields_list, prepared_neg_keywords)
    log_list.append(f"상세(엄격한) 키워드 매칭 결과: {len(keyword_hits_data)}건.")

    df_keyword = pd.DataFrame(keyword_hits_data)
    if not df_keyword.empty:
        df_keyword['relevance_score'] = 100
        df_keyword['relevance_reason'] = '핵심 키워드 매칭 (Core Keyword)'
        df_keyword['collection_method'] = 'Keyword_Core'

    keyword_data_ids = {id(item) for item in keyword_hits_data}
    ai_candidate_data = [item for item in broad_data if id(item) not in keyword_data_ids]

    df_ai_candidates = pd.DataFrame(ai_candidate_data)

    if df_ai_candidates.empty:
        log_list.append(f"\n--- [{log_prefix}] 3단계: AI 분석 ---")
        log_list.append("AI 분석 대상이 없습니다 (모두 핵심 키워드에 매칭되었거나 없음).")
        df_analyzed = pd.DataFrame()
    else:
        log_list.append(f"\n--- [{log_prefix}] 3단계: AI 관련성 분석 시작 (대상: {len(df_ai_candidates)}건) ---")
        df_analyzed = calculate_ai_relevance(gemini_key, df_ai_candidates, data_type, log_list)

        if not df_analyzed.empty and 'relevance_score' in df_analyzed.columns:
            df_analyzed['collection_method'] = df_analyzed.apply(lambda row: 'AI_Analyzed' if row.get('relevance_score', -1) != -1 else 'AI_Error', axis=1)

    final_df = pd.concat([df_keyword, df_analyzed], ignore_index=True)

    if data_type == 'bid' and 'bidNtceNo' in final_df.columns:
         final_df = final_df.drop_duplicates(subset=['bidNtceNo'])

    log_list.append(f"✅ [{log_prefix}] 최종 결과: 총 {len(final_df)}건 수집 및 분석 완료.")
    return final_df


# 메인 실행 함수
# [수정됨] 진단 로직 추가
def run_analysis(search_keywords: set, client: NaraJangteoApiClient, gemini_key: str, start_date: date, end_date: date, auto_expand_keywords: bool = True, min_relevance_score: int = 60):
    
    log = []; all_found_data = {}

    execution_time = datetime.now()
    fmt_date = '%Y%m%d'; fmt_datetime = '%Y%m%d%H%M'
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt_limit = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
    end_dt = min(execution_time, end_dt_limit)

    start_date_str = start_dt.strftime(fmt_date); end_date_str = end_dt.strftime(fmt_date)
    start_datetime_str = start_dt.strftime(fmt_datetime); end_datetime_str = end_dt.strftime(fmt_datetime)

    log.append(f"💡 분석 시작: 검색 기간 {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    if end_dt != end_dt_limit:
        try:
            import pytz
            kst = pytz.timezone('Asia/Seoul')
            current_time_kst = execution_time.astimezone(kst).strftime('%H:%M')
            time_info = f"(KST {current_time_kst})"
        except ImportError:
            current_time_kst = end_dt.strftime('%H:%M')
            time_info = f"({current_time_kst})"
        
        log.append(f"ℹ️ 참고: 종료 시각이 현재 시각{time_info} 기준으로 조정되었습니다.")

    # [중요] 100점 매칭 대상 키워드 확정 (get_strict_match_keywords 로직 적용)
    strict_match_keywords = get_strict_match_keywords(search_keywords)
    log.append(f"💡 키워드 전략 적용: 전체 {len(search_keywords)}개 로드됨 -> 100점 매칭 대상(Core) {len(strict_match_keywords)}개 확정.")
    # 일반 키워드(AI 분석 대상) 확인 로그 추가
    general_keywords_in_use = search_keywords.intersection(GENERAL_KEYWORDS)
    log.append(f"💡 AI 분석 대상 키워드(General): {len(general_keywords_in_use)}개 사용 중.")

    # [신규 추가] 네거티브 키워드 로드 확인 및 진단
    log.append(f"💡 네거티브 키워드 로드됨: 총 {len(NEGATIVE_KEYWORDS)}개.")
    # 특정 키워드가 포함되었는지 확인하여 코드 버전 확인
    diagnostic_check_keywords = ["자동제어", "감시제어", "펌프", "맨홀", "VAV"]
    missing_diag_keys = [kw for kw in diagnostic_check_keywords if kw not in NEGATIVE_KEYWORDS]
    if missing_diag_keys:
        log.append(f"⚠️ [진단 경고] 중요 네거티브 키워드가 누락되었습니다: {missing_diag_keys}. analyzer.py 버전이 최신이 아니거나 로드에 실패했을 수 있습니다. 앱을 재시작해주세요.")
    else:
        log.append(f"✅ [진단 확인] 중요 네거티브 키워드(예: 자동제어, 감시제어)가 정상적으로 포함되어 있습니다.")


    # --- 데이터 수집 및 처리 파이프라인 ---
    # (이하 파이프라인 코드는 기존 코드 유지, 변경 없음)

    # 1. 발주계획
    log.append("\n========== 1. 발주계획 정보 수집 및 분석 ==========")
    current_year = execution_time.year
    target_years = list(set(range(start_date.year, max(end_date.year, current_year + 1) + 1)))
    
    order_plan_params = {'year': target_years}

    all_found_data['order_plan'] = collect_and_analyze(
        client.get_order_plans, order_plan_params, strict_match_keywords, BROAD_KEYWORDS,
        search_fields=['prdctNm', 'dminsttNm'],
        gemini_key=gemini_key, log_list=log, log_prefix="발주계획", data_type="order_plan",
        negative_keywords=NEGATIVE_KEYWORDS
    )
    if not all_found_data['order_plan'].empty:
        upsert_order_plan_data(all_found_data['order_plan'], log)

    # 2. 사전규격
    log.append("\n========== 2. 사전규격 정보 수집 (참조용) ==========")
    pre_standard_params = {'start_date': start_date_str, 'end_date': end_date_str}

    pre_std_raw = fetch_api_data(client.get_pre_standard_specs, pre_standard_params, log, log_prefix="사전규격")
    
    prepared_strict = prepare_keywords(strict_match_keywords)
    prepared_negative = prepare_negative_keywords(NEGATIVE_KEYWORDS)

    pre_std_filtered = []
    if pre_std_raw:
        pre_std_filtered = filter_data(pre_std_raw, prepared_strict, ['prdctClsfcNoNm', 'bsnsNm'], prepared_negative)
        log.append(f"사전규격 필터링 완료: {len(pre_std_filtered)}건.")


    all_found_data['pre_standard'] = pd.DataFrame(pre_std_filtered)
    
    pre_standard_map = {
        r['bfSpecRgstNo']: r for r in pre_std_filtered
        if r.get('bfSpecRgstNo')
    }

    # 3. 입찰공고
    log.append("\n========== 3. 입찰 공고 정보 수집 및 분석 ==========")
    bid_params = {'start_date': start_datetime_str, 'end_date': end_datetime_str}

    bid_df = collect_and_analyze(
        client.get_bid_announcements, bid_params, strict_match_keywords, BROAD_KEYWORDS,
        search_fields=['bidNtceNm', 'ntcelnsttNm'],
        gemini_key=gemini_key, log_list=log, log_prefix="입찰공고", data_type="bid",
        negative_keywords=NEGATIVE_KEYWORDS
    )

    if not bid_df.empty:
        def link_pre_standard(row):
            spec_no = row.get('bfSpecRgstNo')
            if spec_no and spec_no in pre_standard_map:
                return ("확인", spec_no, pre_standard_map[spec_no].get('registDt'))
            return ("해당 없음", None, None)
        bid_df[['prestandard_status', 'prestandard_no', 'prestandard_date']] = bid_df.apply(link_pre_standard, axis=1, result_type='expand')

    all_found_data['bid'] = bid_df
    if not bid_df.empty:
        upsert_project_data(bid_df, 'bid')

    # 4. 낙찰정보
    log.append("\n========== 4. 낙찰 정보 수집 ==========")
    succ_bid_base_params = {'start_date': start_datetime_str, 'end_date': end_datetime_str}
    succ_dfs = []

    for code in ['1','2','3','5']:
        params_with_code = {**succ_bid_base_params, 'bsns_div_cd': code}

        succ_raw = fetch_api_data(client.get_successful_bid_info, params_with_code, log, log_prefix=f"낙찰정보(코드:{code})")
        
        succ_filtered = []
        if succ_raw:
            succ_filtered = filter_data(succ_raw, prepared_strict, ['bidNtceNm', 'ntcelnsttNm'], prepared_negative)

        if succ_filtered:
            succ_dfs.append(pd.DataFrame(succ_filtered))
            log.append(f"낙찰정보(코드:{code}) 필터링 완료: {len(succ_filtered)}건.")

    all_found_data['successful_bid'] = pd.concat(succ_dfs, ignore_index=True) if succ_dfs else pd.DataFrame()
    if not all_found_data['successful_bid'].empty:
        upsert_project_data(all_found_data['successful_bid'], 'successful_bid')


    # 5. 계약정보
    log.append("\n========== 5. 계약 정보 수집 ==========")
    contract_params = {'start_date': start_date_str, 'end_date': end_date_str}

    contract_raw = fetch_api_data(client.get_contract_info, contract_params, log, log_prefix="계약정보")

    contract_filtered = []
    if contract_raw:
        contract_filtered = filter_data(contract_raw, prepared_strict, ['cntrctNm', 'dminsttNm'], prepared_negative)
        log.append(f"계약정보 필터링 완료: {len(contract_filtered)}건.")


    all_found_data['contract'] = pd.DataFrame(contract_filtered)
    
    if not all_found_data['contract'].empty:
        upsert_project_data(all_found_data['contract'], 'contract')


    # --- 보고서 생성 및 후처리 ---
    log.append("\n========== 6. 보고서 생성 및 전략 분석 시작 ==========")
    report_dfs = create_report_data("procurement_data.db", log, min_relevance_score)

    risk_df, report_data_bytes, gemini_report = pd.DataFrame(), None, None

    if report_dfs:
        if "flat" in report_dfs and report_dfs["flat"] is not None:
             risk_df = analyze_project_risk(report_dfs["flat"])

        excel_sheets = {
            "종합 현황 보고서": report_dfs.get("structured"),
            "발주계획 현황": report_dfs.get("order_plan"),
            "리스크 분석": risk_df,
            "발주계획 원본(분석 전체)": all_found_data.get('order_plan'),
            "입찰공고 원본(분석 전체)": all_found_data.get('bid'),
        }
        try:
            report_data_bytes = save_integrated_excel(excel_sheets)
            log.append("✅ 통합 엑셀 보고서 생성 완료.")
        except Exception as e:
            log.append(f"⚠️ 엑셀 파일 생성 중 오류 발생: {e}")

    if report_dfs and "flat" in report_dfs and report_dfs["flat"] is not None:
        gemini_report = get_gemini_analysis(gemini_key, report_dfs["flat"], log)

    if auto_expand_keywords:
        log.append("\n========== 7. AI 기반 키워드 확장 시작 ==========")
        combined_df_list = [df for df in all_found_data.values() if df is not None and not df.empty]
        if combined_df_list:
            combined_df = pd.concat(combined_df_list, ignore_index=True)
            new_keywords = expand_keywords_with_gemini(gemini_key, combined_df, search_keywords, log)
            if new_keywords:
                updated_keywords = search_keywords.union(new_keywords)
                save_keywords(updated_keywords)
                log.append("🎉 키워드 파일이 새롭게 확장되었습니다!")
            else:
                 log.append("ℹ️ 추천된 신규 키워드가 없습니다.")
        else:
             log.append("ℹ️ 키워드 확장을 위한 데이터가 없습니다.")


    return {
        "log": log,
        "risk_df": risk_df,
        "report_file_data": report_data_bytes,
        "report_filename": f"integrated_report_AI_v2_{execution_time.strftime('%Y%m%d_%H%M%S')}.xlsx",
        "gemini_report": gemini_report
    }