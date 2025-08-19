import os
import requests
import pandas as pd
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

# 헬퍼 함수: 안전한 정수 변환 (Robustness 향상)
def safe_int(value):
    try:
        # 빈 문자열 처리 추가
        return int(float(value)) if value is not None and str(value).strip() != "" else None
    except (ValueError, TypeError):
        return None

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
    # years 리스트를 받아서 처리
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
            is_error = False
            if log_list:
                 # 최근 로그 확인 (오류 로그는 ⚠️로 시작하도록 통일됨)
                 is_error = log_list[-1].startswith("⚠️")

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

# [필수] 오류 해결을 위해 포함된 함수
def setup_database():
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()
    
    # 기존 projects 테이블 (입찰공고 ~ 계약)
    cursor.execute("CREATE TABLE IF NOT EXISTS projects (bidNtceNo TEXT PRIMARY KEY, bidNtceNm TEXT, ntcelnsttNm TEXT, presmptPrce INTEGER, bid_status TEXT DEFAULT '공고', bidNtceDate TEXT, sucsfCorpNm TEXT, cntrctAmt INTEGER, cntrctDate TEXT)")
    
    # ALTER TABLE로 컬럼 추가 시 이미 존재하는 경우 오류 발생 방지 (Robustness 향상)
    try:
        # PRAGMA를 사용하여 현재 스키마 확인
        existing_columns = [info[1] for info in cursor.execute("PRAGMA table_info(projects)")]
        if 'prestandard_status' not in existing_columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN prestandard_status TEXT")
        if 'prestandard_no' not in existing_columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN prestandard_no TEXT")
        if 'prestandard_date' not in existing_columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN prestandard_date TEXT")
    except sqlite3.OperationalError as e:
        logging.warning(f"Error altering projects table: {e}")


    # 발주계획 테이블 (order_plans)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_year TEXT,
            category TEXT,        -- 카테고리 (물품/용역/공사)
            dminsttNm TEXT,       -- 수요기관명
            prdctNm TEXT,         -- 품명 (검색 대상 필드)
            asignBdgtAmt INTEGER, -- 배정예산액
            orderInsttNm TEXT,    -- 발주기관명
            orderPlanPrd TEXT,    -- 발주예정시기
            cntrctMthdNm TEXT,    -- 계약방법명
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderPlanPrd)
        )
    """)
    
    conn.commit(); conn.close()

# [필수] 오류 해결을 위해 포함된 함수
def upsert_project_data(df, stage):
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()
    
    for _, r in df.iterrows():
        try:
            if stage == 'bid':
                # safe_int 헬퍼 함수 사용
                cursor.execute("""
                    INSERT OR IGNORE INTO projects 
                    (bidNtceNo, bidNtceNm, ntcelnsttNm, presmptPrce, bidNtceDate, prestandard_status, prestandard_no, prestandard_date) 
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    r.get('bidNtceNo'), r.get('bidNtceNm'), r.get('ntcelnsttNm'), safe_int(r.get('presmptPrce')), r.get('bidNtceDate'),
                    r.get('prestandard_status'), r.get('prestandard_no'), r.get('prestandard_date')
                ))
            elif stage == 'successful_bid':
                cursor.execute("""
                    UPDATE projects SET bid_status='낙찰', sucsfCorpNm=? 
                    WHERE bidNtceNo=? AND (bid_status='공고' OR bid_status IS NULL)
                """, (r.get('sucsfCorpNm'), r.get('bidNtceNo')))
            elif stage == 'contract':
                # 계약 단계에서는 대표업체명(rprsntCorpNm) 사용, safe_int 사용
                cursor.execute("""
                    UPDATE projects SET bid_status='계약완료', sucsfCorpNm=?, cntrctAmt=?, cntrctDate=? 
                    WHERE bidNtceNo=?
                """, (r.get('rprsntCorpNm'), safe_int(r.get('cntrctAmt')), r.get('cntrctCnclsDate'), r.get('bidNtceNo')))
        except Exception as e:
            logging.error(f"Error upserting project data (stage: {stage}): {e} - Data: {r.to_dict()}")
            continue # 개별 레코드 오류는 건너뛰고 계속 진행

    conn.commit(); conn.close()

# [필수] 오류 해결을 위해 포함된 함수
def upsert_order_plan_data(df, log_list):
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()
    initial_count = conn.total_changes
    
    for _, r in df.iterrows():
        try:
            # 'plan_year'는 API 호출 시 데이터에 추가됨, safe_int 사용
            cursor.execute("""
                INSERT OR IGNORE INTO order_plans 
                (plan_year, category, dminsttNm, prdctNm, asignBdgtAmt, orderInsttNm, orderPlanPrd, cntrctMthdNm) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.get('plan_year'),
                r.get('category'),
                r.get('dminsttNm'),
                r.get('prdctNm'),
                safe_int(r.get('asignBdgtAmt')),
                r.get('orderInsttNm'),
                r.get('orderPlanPrd'),
                r.get('cntrctMthdNm')
            ))
        except Exception as e:
            log_list.append(f"⚠️ 경고: 발주계획 DB 삽입 중 오류 발생: {e} - 데이터: {r.to_dict()}")
            continue
            
    conn.commit()
    new_records_count = conn.total_changes - initial_count
    log_list.append(f"발주계획 정보 DB 저장 완료 (신규 {new_records_count}건).")
    conn.close()


# --- 4. AI 분석, 리스크 분석 및 보고서 ---

# 금액 포맷팅 헬퍼 함수
def format_price(x):
    if pd.isna(x):
        return ""
    try:
        # 입력값이 문자열일 경우 쉼표 제거 후 변환 시도
        if isinstance(x, str):
             x = x.replace(',', '')
        return f"{int(float(x)):,}"
    except (ValueError, TypeError):
        return ""

# [필수] 오류 해결을 위해 포함된 함수
def get_gemini_analysis(api_key, df, log_list):
    # AI 분석은 프로젝트 현황(df, flat data)을 기반으로 수행
    if df.empty: log_list.append("AI가 분석할 데이터가 없습니다."); return None
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        log_list.append("Gemini API로 맞춤형 전략 분석 시작...")
        
        # 프롬프트 전달 전 금액 포맷팅 적용 (AI 가독성 향상)
        df_for_prompt = df.copy()
        if 'presmptPrce' in df_for_prompt.columns:
             df_for_prompt['presmptPrce'] = df_for_prompt['presmptPrce'].apply(format_price)
        if 'cntrctAmt' in df_for_prompt.columns:
             df_for_prompt['cntrctAmt'] = df_for_prompt['cntrctAmt'].apply(format_price)
             
        data_for_prompt_str = df_for_prompt[[c for c in ['prestandard_status', 'bidNtceNm', 'bid_status', 'ntcelnsttNm', 'presmptPrce', 'sucsfCorpNm', 'cntrctAmt'] if c in df_for_prompt.columns]].head(30).to_string()
        
        prompt = f"""당신은 '위치 및 동작 인식 기술'을 기반으로 VR/MR/AR/XR 가상환경과 현실공간을 정밀하게 매칭(공간 정합)하는 기술을 보유한, 군/경 훈련 시뮬레이터 전문 기업의 '신사업 전략팀장'입니다. 아래 제공된 나라장터 조달 데이터를 바탕으로, 우리 회사의 기술적 강점을 극대화하고 새로운 사업 기회를 발굴하기 위한 심층 분석 보고서를 작성해주세요. 결과는 반드시 아래 형식에 맞춰 마크다운으로 작성해주세요.\n\n[분석 데이터]\n{data_for_prompt_str}\n\n---\n## 군·경 훈련 시뮬레이션 사업 기회 분석 보고서\n\n### 1. 총평 및 핵심 동향\n(우리 회사의 XR 및 공간 정합 기술과 관련성이 높은 훈련 시뮬레이션, 가상현실, MRO 사업의 증감 추세나, 주목해야 할 발주 기관(육군, 경찰청 등)의 동향을 분석해주세요.)\n\n### 2. 주요 사업 심층 분석\n(우리 회사 기술과의 연관성이 가장 높거나 사업적 가치가 큰 프로젝트 3~5개를 선정하여 아래 표 형식으로 분석해주세요. '분석 및 제언' 항목에는 **우리 회사의 핵심 기술인 '공간 정합', '위치/동작 인식' 기술을 적용할 수 있는 지점이나, 기존 시스템을 고도화할 수 있는 사업 기회**를 중점적으로 작성해주세요.)\n\n| 사업명 | 발주기관 | 추정가격/계약금액 | 진행 상태 | 분석 및 제언 (자사 기술 연계 방안) |\n|---|---|---|---|---|\n| (사업명) | (기관명) | (금액) | (상태) | (예: 이 사업은 CQB 훈련 시뮬레이터로, **현실 공간과 가상 훈련 시나리오를 정밀하게 매칭하는 우리 기술**이 핵심 경쟁력이 될 수 있음.) |\n\n### 3. 기술 연계 가능 키워드\n(데이터에서 식별된 키워드 중, 우리 회사의 '위치/동작 인식' 및 'XR 가시화' 기술과 직접적으로 연결될 수 있는 핵심 키워드를 5개 이상 선정하여 불렛 포인트로 나열해주세요.)\n\n### 4. 차기 사업 전략 제언\n(위 분석을 종합하여, 우리 회사가 다음 분기에 집중해야 할 사업 영역, 기술 고도화 방향 등에 대한 구체적인 실행 전략을 1~2가지 제언해주세요.)"""
        response = model.generate_content(prompt); log_list.append("Gemini 맞춤형 전략 분석 완료.")
        return response.text
    except Exception as e: 
        log_list.append(f"⚠️ Gemini API 호출 중 오류 발생: {e}")
        logging.error(f"Gemini API Error: {e}")
        return None

# [필수] 오류 해결을 위해 포함된 함수
def expand_keywords_with_gemini(api_key, df, existing_keywords, log_list):
    # 키워드 확장은 수집된 모든 데이터(df)를 기반으로 수행
    if df.empty: log_list.append("키워드 확장을 위한 분석 데이터가 없습니다."); return set()
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        log_list.append("Gemini API로 지능형 키워드 확장 시작...")
        # 발주계획(prdctNm) 및 사전규격(prdctClsfcNoNm)도 포함하여 분석 대상 확대
        project_titles = pd.concat([df[col] for col in ['bidNtceNm', 'cntrctNm', 'prdctClsfcNoNm', 'prdctNm'] if col in df.columns]).dropna().unique()
        project_titles_str = '\n- '.join(project_titles[:50]) # 프롬프트 길이 고려
        
        prompt = f"""당신은 군/경 훈련 시뮬레이터 전문 기업의 조달 정보 분석가입니다. 우리 회사는 '위치/동작 인식', 'XR 공간 정합' 기술을 보유하고 있습니다. 아래는 현재 우리가 사용중인 검색 키워드와, 최근 조달 시스템에서 발견된 사업명/품명 목록입니다. 이 정보를 바탕으로, 우리 회사의 기술과 관련성이 높으면서도 기존 키워드에 없는 **새로운 검색 키워드를 5~10개 추천**해주세요. 결과는 다른 설명 없이, 쉼표(,)로 구분된 키워드 목록으로만 제공해주세요.\n\n[기존 키워드]\n{', '.join(sorted(list(existing_keywords)))}\n\n[최근 발견된 사업명/품명]\n- {project_titles_str}\n\n[추천 키워드]"""
        response = model.generate_content(prompt)
        new_keywords = {k.strip() for k in response.text.strip().split(',') if k.strip() and len(k.strip()) > 1}
        log_list.append(f"Gemini가 추천한 신규 키워드: {new_keywords}")
        return new_keywords
    except Exception as e: 
        log_list.append(f"⚠️ Gemini 키워드 확장 중 오류 발생: {e}")
        return set()

# [필수] 오류 해결을 위해 포함된 함수
def analyze_project_risk(df: pd.DataFrame) -> pd.DataFrame:
    # 리스크 분석은 flat data(금액이 숫자형)를 기반으로 수행
    ongoing_df = df[df['bid_status'].isin(['공고', '낙찰'])].copy()
    if ongoing_df.empty: return pd.DataFrame()
    
    ongoing_df['score'] = 0
    ongoing_df['risk_reason'] = ''
    # 날짜 컬럼을 datetime 객체로 변환
    ongoing_df['bidNtceDate_dt'] = pd.to_datetime(ongoing_df['bidNtceDate'], errors='coerce')
    
    # 분석 시점 기준 시간
    current_time = datetime.now()

    for index, row in ongoing_df.iterrows():
        reasons = []
        score = 0
        
        # 리스크 요인 1: 공고 후 30일 경과 (분석 시점 기준)
        if pd.notna(row['bidNtceDate_dt']) and row['bid_status'] == '공고':
            days_elapsed = (current_time - row['bidNtceDate_dt']).days
            if days_elapsed > 30:
                score += 5
                reasons.append(f'공고 후 {days_elapsed}일 경과')
        
        # 리스크 요인 2: 사전규격 미공개
        if row.get('prestandard_status') == '해당 없음':
            score += 3
            reasons.append('사전규격 미공개')
        
        # 리스크 요인 3: 소규모 사업 (5천만원 미만)
        # flat data이므로 금액은 숫자형임
        price = row.get('presmptPrce')
        # 숫자형인지 확인 후 비교
        if pd.notna(price) and isinstance(price, (int, float)) and 0 < price < 50000000:
            score += 2
            reasons.append('소규모 사업 (5천만원 미만)')
        
        ongoing_df.loc[index, 'score'] = score
        ongoing_df.loc[index, 'risk_reason'] = ', '.join(reasons)
    
    # 리스크 등급 산정
    ongoing_df['risk_level'] = ongoing_df['score'].apply(lambda s: '높음' if s >= 7 else ('보통' if s >= 4 else '낮음'))
    
    # 보고서용 테이블 생성 및 정렬
    risk_table = ongoing_df[['bidNtceNm', 'ntcelnsttNm', 'bid_status', 'risk_level', 'risk_reason']]
    return risk_table.rename(columns={'bidNtceNm':'사업명','ntcelnsttNm':'발주기관','bid_status':'진행 상태','risk_level':'리스크 등급','risk_reason':'주요 리스크'}).sort_values(by='리스크 등급',key=lambda x:x.map({'높음':0,'보통':1,'낮음':2}))

# [필수] 오류 해결을 위해 포함된 함수
def create_report_data(db_path, keywords, log_list):
    log_list.append("DB에서 최종 데이터 조회 중...")
    conn = sqlite3.connect(db_path)
    report_data = {} # 결과를 담을 딕셔너리

    try:
        # 1. 프로젝트(입찰공고 이후) 데이터 처리
        try:
            # bidNtceDate 기준으로 내림차순 정렬하여 최신순으로 조회
            all_projects_df = pd.read_sql_query("SELECT * FROM projects ORDER BY bidNtceDate DESC", conn)
        except pd.errors.DatabaseError as e:
            log_list.append(f"⚠️ 프로젝트 DB 조회 중 오류: {e}")
            all_projects_df = pd.DataFrame()

        
        if not all_projects_df.empty:
            # 키워드 필터링
            flat_df = all_projects_df[all_projects_df['bidNtceNm'].str.contains('|'.join(keywords),case=False,na=False)].copy()
            
            if not flat_df.empty:
                # [중요] 분석용 원본 데이터 저장 (금액: 숫자형)
                report_data["flat"] = flat_df.copy()

                # 보고서용 데이터 포맷팅 시작 (이후 flat_df는 보고서용으로 사용)
                # 날짜 포맷팅 (YYYY-MM-DD)
                for col in ['prestandard_date','bidNtceDate','cntrctDate']:
                    if col in flat_df.columns: 
                        flat_df[col] = pd.to_datetime(flat_df[col],errors='coerce').dt.strftime('%Y-%m-%d')
                
                # 금액 포맷팅 (금액: 문자열)
                for col in ['presmptPrce','cntrctAmt']:
                    if col in flat_df.columns: 
                        flat_df[col] = flat_df[col].apply(format_price)

                # 구조화된 데이터프레임 (MultiIndex) 생성
                structured_columns = {
                    ('프로젝트 개요','사업명'):flat_df.get('bidNtceNm'), 
                    ('프로젝트 개요','발주기관'):flat_df.get('ntcelnsttNm'), 
                    ('진행 현황','종합 상태'):flat_df.get('bid_status'), 
                    ('진행 현황','낙찰/계약사'):flat_df.get('sucsfCorpNm'), 
                    ('사전 규격 정보','공개 상태'):flat_df.get('prestandard_status'), 
                    ('사전 규격 정보','공개일'):flat_df.get('prestandard_date'), 
                    ('입찰 공고 정보','공고일'):flat_df.get('bidNtceDate'), 
                    ('입찰 공고 정보','추정가격'):flat_df.get('presmptPrce'), 
                    ('계약 체결 정보','계약일'):flat_df.get('cntrctDate'), 
                    ('계약 체결 정보','계약금액'):flat_df.get('cntrctAmt'), 
                    ('참조 번호','사전규격번호'):flat_df.get('prestandard_no'), 
                    ('참조 번호','입찰공고번호'):flat_df.get('bidNtceNo')
                }
                report_data["structured"] = pd.DataFrame(structured_columns)
                log_list.append("프로젝트 현황 보고서 데이터 생성 완료.")

        # 2. 발주계획 데이터 처리
        try:
            # created_at 기준으로 내림차순 정렬하여 최신 데이터 조회
            all_order_plans_df = pd.read_sql_query("SELECT * FROM order_plans ORDER BY created_at DESC", conn)
        except pd.errors.DatabaseError as e:
             log_list.append(f"⚠️ 발주계획 DB 조회 중 오류: {e}")
             all_order_plans_df = pd.DataFrame()

        if not all_order_plans_df.empty:
             # 키워드 필터링 (품명 기준 - prdctNm)
            order_plan_df = all_order_plans_df[all_order_plans_df['prdctNm'].str.contains('|'.join(keywords),case=False,na=False)].copy()

            if not order_plan_df.empty:
                # 중복 제거: (연도, 카테고리, 기관명, 품명) 기준으로 가장 최신 데이터만 남김
                # 이미 created_at DESC로 정렬되었으므로 keep='first' 사용
                order_plan_df = order_plan_df.drop_duplicates(subset=['plan_year', 'category', 'dminsttNm', 'prdctNm'], keep='first')

                # 금액 포맷팅 (원본 금액은 유지하고 포맷팅된 컬럼 추가)
                order_plan_df['asignBdgtAmt_formatted'] = order_plan_df['asignBdgtAmt'].apply(format_price)
                
                # 보고서용 컬럼명 변경 및 선택
                order_plan_report_df = order_plan_df[[
                    'plan_year', 'category', 'dminsttNm', 'prdctNm', 'asignBdgtAmt_formatted', 'orderPlanPrd', 'cntrctMthdNm'
                ]].rename(columns={
                    'plan_year': '년도',
                    'category': '구분(물품/용역/공사)',
                    'dminsttNm': '수요기관명',
                    'prdctNm': '품명 (사업명)',
                    'asignBdgtAmt_formatted': '배정예산액',
                    'orderPlanPrd': '발주예정시기',
                    'cntrctMthdNm': '계약방법'
                })
                # 최종 정렬: 예산액 기준 내림차순 (문자열로 포맷팅된 금액을 숫자로 변환하여 정렬)
                report_data["order_plan"] = order_plan_report_df.sort_values(by='배정예산액', ascending=False, key=lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))
                log_list.append("발주계획 현황 보고서 데이터 생성 완료.")

        if not report_data:
             log_list.append("DB에 키워드에 해당하는 데이터(프로젝트 또는 발주계획)가 없습니다.")
             return None
             
        return report_data

    except Exception as e: 
        log_list.append(f"⚠️ 보고서 데이터 생성 중 예상치 못한 오류 발생: {e}")
        logging.exception(e)
        return None
    finally: 
        conn.close()


# --- 5. 메인 실행 함수 ---
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
    # 시작 연도부터 종료 연도까지 모든 연도 리스트 생성
    target_years = list(range(start_date.year, end_date.year + 1))
    
    # API 호출 파라미터 (year에 연도 리스트 전달)
    order_plan_params = {'year': target_years}
    
    # 데이터 조회 및 필터링
    all_found_data['order_plan'] = search_and_process(
        client.get_order_plans, order_plan_params, search_keywords, 'prdctNm', log, 
        log_prefix="발주계획"
    )
    # DB 저장
    if not all_found_data['order_plan'].empty:
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
        # 리스크 분석 (DB에서 가져온 원본 'flat' 데이터 사용 - 숫자형)
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
    if gemini_key:
        if report_dfs:
            # AI 전략 분석 (DB에서 가져온 원본 'flat' 데이터 사용 - 숫자형)
            if "flat" in report_dfs and report_dfs["flat"] is not None:
                gemini_report = get_gemini_analysis(gemini_key, report_dfs["flat"], log)
            
            # 키워드 확장 (API로 수집된 모든 데이터 통합)
            if auto_expand_keywords:
                combined_df_list = [df for df in all_found_data.values() if df is not None and not df.empty]
                if combined_df_list:
                    combined_df = pd.concat(combined_df_list, ignore_index=True)
                    new_keywords = expand_keywords_with_gemini(gemini_key, combined_df, search_keywords, log)
                    if new_keywords:
                        updated_keywords = search_keywords.union(new_keywords)
                        save_keywords(updated_keywords)
                        log.append("🎉 키워드 파일이 새롭게 확장되었습니다!")
        else:
             log.append("ℹ️ 분석할 데이터가 없어 AI 분석 및 키워드 확장을 생략합니다.")

    
    # 최종 결과 반환
    return {
        "log": log, 
        "risk_df": risk_df, 
        "report_file_data": report_data_bytes, 
        "report_filename": f"integrated_report_{execution_time.strftime('%Y%m%d_%H%M%S')}.xlsx", 
        "gemini_report": gemini_report
    }