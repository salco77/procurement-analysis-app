import os
import requests
import pandas as pd
# [수정] date 타입 힌팅 추가
from datetime import datetime, timedelta, date
import urllib.parse
import time
import sqlite3
import google.generativeai as genai
import io
from itertools import groupby
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# --- 1. 키워드 관리 ---
KEYWORD_FILE = "keywords.txt"
INITIAL_KEYWORDS = ["지뢰","드론","시뮬레이터","시뮬레이션","전차","유지보수","MRO","항공","가상현실","증강현실","훈련","VR","AR","MR","XR","대테러","소부대","CQB","특전사","경찰청"]

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
def save_integrated_excel(data_frames: dict) -> bytes:
    """여러 데이터프레임을 통합 엑셀로 저장합니다. MultiIndex 및 발주계획 스타일링 지원."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            # 발주계획 시트 스타일 정의 (연한 노란색)
            order_plan_header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#FFF2CC', 'border': 1})

            for sheet_name, df_data in data_frames.items():
                if df_data is None or df_data.empty: continue
                
                # MultiIndex 처리 로직 (종합 현황 보고서)
                if isinstance(df_data.columns, pd.MultiIndex) and df_data.columns.nlevels == 2:
                    worksheet = workbook.add_worksheet(sheet_name)
                    writer.sheets[sheet_name] = worksheet
                    header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#D3D3D3', 'border': 1})
                    
                    # 1. 헤더 작성 (병합 포함)
                    col_idx = 0
                    level0_headers = df_data.columns.get_level_values(0)
                    level1_headers = df_data.columns.get_level_values(1)

                    for header, group in groupby(level0_headers):
                        span = len(list(group))
                        if span > 1:
                            worksheet.merge_range(0, col_idx, 0, col_idx + span - 1, str(header), header_format)
                        else:
                            worksheet.write(0, col_idx, str(header), header_format)
                        for i in range(span):
                            worksheet.write(1, col_idx + i, str(level1_headers[col_idx + i]), header_format)
                        col_idx += span

                    # 2. 데이터 작성
                    start_row = 2
                    def clean_data(x): return x if pd.notna(x) else ""
                    
                    # Pandas 버전 호환성 처리
                    try:
                        if hasattr(df_data, 'map'):
                             data_to_write = df_data.map(clean_data).values.tolist()
                        else:
                             # applymap (구버전 호환성)
                             data_to_write = df_data.applymap(clean_data).values.tolist()
                    except Exception:
                        # Fallback
                        data_to_write = df_data.fillna("").values.tolist()

                    for row_data in data_to_write:
                        worksheet.write_row(start_row, 0, row_data)
                        start_row += 1
                    
                    # 3. 컬럼 너비 자동 조정
                    for i in range(len(df_data.columns)):
                        header_len = max(len(str(level0_headers[i])), len(str(level1_headers[i])))
                        try:
                            data_max_len = max((len(str(row[i])) for row in data_to_write), default=0)
                        except Exception:
                            data_max_len = 10
                        max_len = max(header_len, data_max_len)
                        worksheet.set_column(i, i, min(60, max(10, max_len + 2)))

                # 발주계획 현황 시트 처리 (스타일 적용)
                elif sheet_name == "발주계획 현황":
                    df_data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
                    worksheet = writer.sheets[sheet_name]
                    
                    # 헤더 작성 및 스타일 적용
                    for col_num, value in enumerate(df_data.columns.values):
                        worksheet.write(0, col_num, value, order_plan_header_format)
                    
                    # 컬럼 너비 자동 조정
                    for i, col in enumerate(df_data.columns):
                        try:
                            max_len = max(df_data[col].astype(str).map(len).max(), len(str(col)))
                        except Exception:
                            max_len = 15
                        worksheet.set_column(i, i, min(60, max(10, max_len + 2)))

                else: 
                    # 일반 DF 처리
                    df_data.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        logging.error(f"Error saving Excel file: {e}")
        raise

    return output.getvalue()


class NaraJangteoApiClient:
    def __init__(self, service_key: str):
        if not service_key: raise ValueError("서비스 키는 필수입니다.")
        self.service_key = service_key
        self.base_url_std = "http://apis.data.go.kr/1230000/ao/PubDataOpnStdService"
        self.base_url_plan = "http://apis.data.go.kr/1230000/ao/OrderPlanSttusService"

    def _make_request(self, base_url: str, endpoint: str, params: dict, log_list: list):
        try:
            url = f"{base_url}/{endpoint}?{urllib.parse.urlencode({'ServiceKey': self.service_key, 'pageNo': 1, 'numOfRows': 999, 'type': 'json', **params}, safe='/%')}"
            # 타임아웃을 90초로 설정
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            log_list.append(f"⚠️ 네트워크 요청 실패 ({endpoint}): {e}")
            return []
        except ValueError:
            log_list.append(f"⚠️ API 응답이 올바른 JSON 형식이 아닙니다 ({endpoint}): {response.text[:200]}...")
            return []

        response_data = data.get('response', {})
        header = response_data.get('header', {})
        if header.get('resultCode') != '00':
            log_list.append(f"⚠️ API 오류 ({endpoint}): {header.get('resultMsg', '오류 메시지 없음')}")
            return []
        
        body = response_data.get('body', {})
        if isinstance(body, list): return body
        if isinstance(body, dict):
            return body.get('items', [])
        return []

    # --- 발주계획현황서비스 (OrderPlanSttusService) ---
    # [수정] years 리스트를 받아서 처리 (year 파라미터명은 유지하되 리스트 처리 로직 추가)
    def get_order_plans(self, year, log_list):
        # 단일 연도(str/int) 입력 시 리스트로 변환
        years = [str(year)] if isinstance(year, (str, int)) else [str(y) for y in year]

        endpoints = {
            '물품': 'getOrderPlanSttusListThng',
            '용역': 'getOrderPlanSttusListServc',
            '공사': 'getOrderPlanSttusListConst'
        }
        all_plans = []
        
        # 연도별로 반복 호출
        for current_year in years:
            params = {'year': current_year}
            log_list.append(f"[{current_year}년도] 발주계획 조회 시작...")
            year_plans_count = 0
            for category, endpoint in endpoints.items():
                log_list.append(f"  - 카테고리: {category} 조회 중...")
                plans = self._make_request(self.base_url_plan, endpoint, params, log_list)
                if plans:
                    # 데이터에 카테고리 및 연도 정보 추가 (DB 저장 시 활용)
                    for plan in plans:
                        plan['category'] = category
                        plan['plan_year'] = current_year
                    all_plans.extend(plans)
                    year_plans_count += len(plans)
            log_list.append(f"[{current_year}년도] 총 {year_plans_count}건 수신.")
        
        log_list.append(f"발주계획 전체 총 {len(all_plans)}건 수신 완료.")
        return all_plans

    # --- 공공데이터개방표준서비스 (PubDataOpnStdService) ---
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


def search_and_process(fetch_function, params, keywords, search_field, log_list, log_prefix=""):
    """API 호출 및 키워드 필터링을 수행하는 공통 함수"""
    # 발주계획은 fetch_function 내부에서 로그를 시작하므로 여기서는 생략
    if log_prefix != "발주계획":
        log_list.append(f"[{log_prefix}] 조회 시작...")
    
    # fetch_function에 params 딕셔너리를 언패킹하여 전달
    raw_data = fetch_function(log_list=log_list, **params)
    
    # 발주계획은 함수 내에서 로그를 상세히 기록했으므로 중복 방지
    if log_prefix != "발주계획":
        if not raw_data: 
            # API/네트워크 오류(⚠️)가 발생하지 않았을 경우에만 "데이터 없음" 로그 기록
            is_error = any("⚠️" in log for log in log_list[-2:]) # 최근 2개 로그 확인
            if not is_error:
                log_list.append("조회된 데이터가 없습니다.")
            return pd.DataFrame()
        log_list.append(f"총 {len(raw_data)}건 수신.")

    if not raw_data: return pd.DataFrame()

    log_list.append(f"키워드 필터링 시작 (검색 필드: {search_field})...")
    # 키워드 필터링 로직
    filtered_data = [
        item for item in raw_data 
        if isinstance(item, dict) and item.get(search_field) and 
        # 검색 필드 값을 문자열로 변환하여 안정성 확보
        any(keyword.lower() in str(item[search_field]).lower() for keyword in keywords)
    ]
    log_list.append(f"필터링 후 {len(filtered_data)}건 발견.")
    return pd.DataFrame(filtered_data)

# --- 3. 데이터베이스 ---
# (setup_database, upsert_project_data 함수는 이전 답변과 동일하여 생략)
# 주의: 실제 파일에는 포함되어야 합니다.

# [수정] year 인자 제거, DataFrame의 'plan_year' 컬럼 사용
def upsert_order_plan_data(df, log_list):
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()
    initial_count = conn.total_changes
    
    for _, r in df.iterrows():
        try:
            # 배정예산액(asignBdgtAmt) 정수형 변환 시도
            try:
                budget = int(float(r.get('asignBdgtAmt'))) if r.get('asignBdgtAmt') else None
            except (ValueError, TypeError):
                budget = None

            # 'plan_year'는 API 호출 시 데이터에 추가됨
            cursor.execute("""
                INSERT OR IGNORE INTO order_plans 
                (plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderInsttNm, orderPlanPrd, cntrctMthdNm) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.get('plan_year'), # API 호출 시 추가된 연도 정보 사용
                r.get('category'),
                r.get('dminsttNm'),
                r.get('prdctNm'),
                budget,
                r.get('orderInsttNm'),
                r.get('orderPlanPrd'),
                r.get('cntrctMthdNm')
            ))
        except Exception as e:
            log_list.append(f"⚠️ 경고: 발주계획 DB 삽입 중 오류 발생: {e} - 데이터: {r.to_dict()}")
            # 개별 레코드 오류는 건너뛰고 계속 진행
            continue
            
    conn.commit()
    new_records_count = conn.total_changes - initial_count
    log_list.append(f"발주계획 정보 DB 저장 완료 (신규 {new_records_count}건).")
    conn.close()


# --- 4. AI 분석, 리스크 분석 및 보고서 ---
# (format_price, get_gemini_analysis, expand_keywords_with_gemini, analyze_project_risk, create_report_data 함수는 이전 답변과 동일하여 생략)
# 주의: 실제 파일에는 포함되어야 합니다.


# --- 5. 메인 실행 함수 (전면 수정됨) ---
# [수정] start_date, end_date를 입력받도록 변경 (Streamlit에서 date 객체로 전달됨)
def run_analysis(search_keywords: set, client: NaraJangteoApiClient, gemini_key: str, start_date: date, end_date: date, auto_expand_keywords: bool = True):
    log = []; all_found_data = {}
    
    # 실행 시점 기록
    execution_time = datetime.now() 

    # --- 날짜 형식 정의 및 변환 ---
    fmt_date = '%Y%m%d'
    fmt_datetime = '%Y%m%d%H%M'

    # 시작일 (00시 00분) 설정
    start_dt = datetime.combine(start_date, datetime.min.time())
    
    # 종료일 (23시 59분) 설정 및 현재 시간과 비교 (API는 미래 날짜 조회 불가)
    end_dt_limit = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
    end_dt = min(execution_time, end_dt_limit)

    # 포맷팅된 문자열 생성
    start_date_str = start_dt.strftime(fmt_date)
    end_date_str = end_dt.strftime(fmt_date)
    start_datetime_str = start_dt.strftime(fmt_datetime)
    end_datetime_str = end_dt.strftime(fmt_datetime)

    log.append(f"💡 분석 시작: 검색 기간 {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    if end_dt != end_dt_limit:
        log.append(f"ℹ️ 참고: 종료 시각이 현재 시각({end_dt.strftime('%H:%M')})으로 조정되었습니다.")


    # --- 데이터 수집 및 처리 파이프라인 ---

    # 1. 발주계획 (연간)
    log.append("\n--- 1. 발주계획 정보 조회 시작 ---")
    # [수정] 시작 연도부터 종료 연도까지 모든 연도 리스트 생성
    target_years = list(range(start_date.year, end_date.year + 1))
    
    # API 호출 파라미터 (year에 연도 리스트 전달 - Client가 리스트 처리 지원하도록 수정됨)
    order_plan_params = {'year': target_years}
    
    # 데이터 조회 및 필터링
    all_found_data['order_plan'] = search_and_process(
        client.get_order_plans, order_plan_params, search_keywords, 'prdctNm', log, 
        log_prefix="발주계획"
    )
    # DB 저장
    if not all_found_data['order_plan'].empty:
        # 개선된 upsert 함수 호출 (데이터프레임 내 'plan_year' 사용)
        upsert_order_plan_data(all_found_data['order_plan'], log)

    # 2. 사전규격 (지정 기간)
    log.append("\n--- 2. 사전규격 정보 조회 시작 ---")
    pre_standard_params = {
        'start_date': start_date_str, 
        'end_date': end_date_str
    }
    all_found_data['pre_standard'] = search_and_process(
        client.get_pre_standard_specs, pre_standard_params, search_keywords, 'prdctClsfcNoNm', log,
        log_prefix=f"사전규격({start_date_str}~{end_date_str})"
    )
    # 사전규격 번호 매핑 딕셔너리 생성
    pre_standard_map = {
        r['bfSpecRgstNo']: r for _, r in all_found_data['pre_standard'].iterrows() 
        if 'bfSpecRgstNo' in r and r['bfSpecRgstNo']
    } if not all_found_data['pre_standard'].empty else {}
    
    # 3. 입찰공고 (지정 기간, 시간 포함)
    log.append("\n--- 3. 입찰 공고 정보 조회 시작 ---")
    bid_params = {
        'start_date': start_datetime_str, 
        'end_date': end_datetime_str
    }
    bid_df = search_and_process(
        client.get_bid_announcements, bid_params, search_keywords, 'bidNtceNm', log,
        log_prefix=f"입찰공고({start_datetime_str}~{end_datetime_str})"
    )
    
    # 입찰공고와 사전규격 연결
    if not bid_df.empty:
        def link_pre_standard(row):
            spec_no = row.get('bfSpecRgstNo')
            if spec_no and spec_no in pre_standard_map:
                return ("확인", spec_no, pre_standard_map[spec_no].get('registDt'))
            return ("해당 없음", None, None)
        bid_df[['prestandard_status', 'prestandard_no', 'prestandard_date']] = bid_df.apply(link_pre_standard, axis=1, result_type='expand')
    
    all_found_data['bid'] = bid_df
    upsert_project_data(bid_df, 'bid')

    # 4. 낙찰정보 (지정 기간, 시간 포함)
    log.append("\n--- 4. 낙찰 정보 조회 시작 ---")
    succ_bid_base_params = {
        'start_date': start_datetime_str, 
        'end_date': end_datetime_str
    }
    succ_dfs = []
    # 업무 구분 코드별 조회
    for code in ['1','2','3','5']:
        params_with_code = {**succ_bid_base_params, 'bsns_div_cd': code}
        df = search_and_process(
            client.get_successful_bid_info, params_with_code, search_keywords, 'bidNtceNm', log,
            log_prefix=f"낙찰정보(코드:{code})"
        )
        if not df.empty:
            succ_dfs.append(df)
            
    all_found_data['successful_bid'] = pd.concat(succ_dfs, ignore_index=True) if succ_dfs else pd.DataFrame()
    if not all_found_data['successful_bid'].empty: 
        upsert_project_data(all_found_data['successful_bid'], 'successful_bid')

    # 5. 계약정보 (지정 기간)
    log.append("\n--- 5. 계약 정보 조회 시작 ---")
    contract_params = {
        'start_date': start_date_str, 
        'end_date': end_date_str
    }
    all_found_data['contract'] = search_and_process(
        client.get_contract_info, contract_params, search_keywords, 'cntrctNm', log,
        log_prefix=f"계약정보({start_date_str}~{end_date_str})"
    )
    if not all_found_data['contract'].empty: 
        upsert_project_data(all_found_data['contract'], 'contract')
    
    # --- 보고서 생성 및 후처리 ---
    log.append("\n--- 6. 보고서 생성 및 분석 시작 ---")
    # 보고서는 DB에 누적된 데이터 중 키워드에 해당하는 모든 것을 기반으로 생성
    report_dfs = create_report_data("procurement_data.db", list(search_keywords), log)
    risk_df, report_data_bytes, gemini_report = pd.DataFrame(), None, None

    # 데이터가 하나라도 존재하면 엑셀 생성 진행
    if report_dfs:
        # 리스크 분석
        if "flat" in report_dfs and report_dfs["flat"] is not None:
             risk_df = analyze_project_risk(report_dfs["flat"])

        # 엑셀 시트 구성 (순서 지정)
        excel_sheets = {
            "종합 현황 보고서": report_dfs.get("structured"),
            "발주계획 현황": report_dfs.get("order_plan"),
            "리스크 분석": risk_df,
            "발주계획 원본": all_found_data.get('order_plan'),
            "사전규격 원본": all_found_data.get('pre_standard'),
            "입찰공고 원본": all_found_data.get('bid'),
            "낙찰정보 원본": all_found_data.get('successful_bid'),
            "계약정보 원본": all_found_data.get('contract')
        }
        # 엑셀 파일 생성
        try:
            report_data_bytes = save_integrated_excel(excel_sheets)
            log.append("✅ 통합 엑셀 보고서 생성 완료.")
        except Exception as e:
            log.append(f"⚠️ 엑셀 파일 생성 중 오류 발생: {e}")

    # AI 분석 및 키워드 확장
    if gemini_key and report_dfs:
        # AI 전략 분석
        if "flat" in report_dfs and report_dfs["flat"] is not None:
            gemini_report = get_gemini_analysis(gemini_key, report_dfs["flat"], log)
        
        # 키워드 확장
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
        "report_filename": f"integrated_report_{execution_time.strftime('%Y%m%d_%H%M%S')}.xlsx", 
        "gemini_report": gemini_report
    }

# ====================================================================================
# 참고: 아래 함수들은 이전 답변의 코드를 그대로 사용합니다. (지면 관계상 생략)
# 실제 analyzer.py 파일에는 아래 함수들이 모두 포함되어야 합니다.
# ====================================================================================
# def setup_database(): ...
# def upsert_project_data(df, stage): ...
# def format_price(x): ...
# def get_gemini_analysis(api_key, df, log_list): ...
# def expand_keywords_with_gemini(api_key, df, existing_keywords, log_list): ...
# def analyze_project_risk(df: pd.DataFrame) -> pd.DataFrame: ...
# def create_report_data(db_path, keywords, log_list): ...