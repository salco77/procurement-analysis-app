import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse
import time
import sqlite3
import google.generativeai as genai
import io
from itertools import groupby # [추가] 헤더 병합을 위해 필요

# --- 1. 키워드 관리 ---
KEYWORD_FILE = "keywords.txt"
INITIAL_KEYWORDS = ["지뢰","드론","시뮬레이터","시뮬레이션","전차","유지보수","MRO","항공","가상현실","증강현실","훈련","VR","AR","MR","XR","대테러","소부대","CQB","특전사","경찰청"]

def load_keywords(initial_keywords: list) -> set:
    if not os.path.exists(KEYWORD_FILE):
        save_keywords(set(initial_keywords))
        return set(initial_keywords)
    with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def save_keywords(keywords: set):
    with open(KEYWORD_FILE, 'w', encoding='utf-8') as f:
        for keyword in sorted(list(keywords)):
            f.write(keyword + '\n')

# --- 2. 유틸리티 및 API 클라이언트 (수정됨) ---
def save_integrated_excel(data_frames: dict) -> bytes:
    """[핵심 수정] 여러 데이터프레임을 통합 엑셀로 저장합니다. MultiIndex 오류를 해결하고 헤더 병합 및 자동 너비 맞춤을 지원합니다."""
    output = io.BytesIO()
    # 수동 제어를 위해 xlsxwriter 엔진 사용
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        for sheet_name, df_data in data_frames.items():
            if df_data is None or df_data.empty: continue
            
            # --- [오류 해결 및 기능 개선] MultiIndex 처리 로직 전면 수정 ---
            # create_report_data는 2단계 MultiIndex를 생성하므로 이에 맞춰 최적화된 로직 적용
            if isinstance(df_data.columns, pd.MultiIndex) and df_data.columns.nlevels == 2:
                # Pandas to_excel 대신 xlsxwriter로 직접 작성하여 NotImplementedError 회피
                
                # 워크시트 생성 및 등록
                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet

                header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center', 'fg_color': '#D3D3D3', 'border': 1})
                
                # 1. 헤더 작성 (2단계)
                col_idx = 0
                level0_headers = df_data.columns.get_level_values(0)
                level1_headers = df_data.columns.get_level_values(1)

                # Level 0 헤더 그룹별로 반복 처리 (셀 병합)
                for header, group in groupby(level0_headers):
                    span = len(list(group))
                    if span > 1:
                        # 셀 병합하여 Level 0 헤더 작성 (0행)
                        # merge_range(first_row, first_col, last_row, last_col, data, format)
                        worksheet.merge_range(0, col_idx, 0, col_idx + span - 1, str(header), header_format)
                    else:
                        # 단일 셀에 Level 0 헤더 작성
                        worksheet.write(0, col_idx, str(header), header_format)
                    
                    # Level 1 헤더 작성 (1행)
                    for i in range(span):
                        worksheet.write(1, col_idx + i, str(level1_headers[col_idx + i]), header_format)
                    
                    col_idx += span

                # 2. 데이터 작성 (2행부터 시작)
                start_row = 2
                
                # NaN/NaT 값을 빈 문자열로 변환하고 리스트로 변환
                # Pandas 버전에 따른 호환성 처리 (applymap vs map)
                def clean_data(x):
                    return x if pd.notna(x) else ""

                try:
                    # Pandas 최신 버전 (map)
                    data_to_write = df_data.map(clean_data).values.tolist()
                except AttributeError:
                    # Pandas 구버전 호환성 (applymap)
                    data_to_write = df_data.applymap(clean_data).values.tolist()

                for row_data in data_to_write:
                    worksheet.write_row(start_row, 0, row_data)
                    start_row += 1
                
                # 3. 컬럼 너비 자동 조정 (가독성 향상)
                for i in range(len(df_data.columns)):
                    # 헤더 길이와 데이터 길이를 고려하여 최대 길이 계산
                    header_len = max(len(str(level0_headers[i])), len(str(level1_headers[i])))
                    data_max_len = max((len(str(row[i])) for row in data_to_write), default=0)
                    max_len = max(header_len, data_max_len)
                    # 너비 설정 (최소 10, 최대 60으로 제한)
                    worksheet.set_column(i, i, min(60, max(10, max_len + 2)))

            else: 
                # 일반 DF 또는 예상치 못한 MultiIndex 레벨 처리
                df_data.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()


class NaraJangteoApiClient:
    # (이하 클래스 및 함수들은 변경 없음)
    def __init__(self, service_key: str):
        if not service_key: raise ValueError("서비스 키는 필수입니다.")
        self.service_key = service_key
        self.base_url = "http://apis.data.go.kr/1230000/ao/PubDataOpnStdService"
    def _make_request(self, endpoint: str, params: dict, log_list: list):
        try:
            url = f"{self.base_url}/{endpoint}?{urllib.parse.urlencode({'ServiceKey': self.service_key, 'pageNo': 1, 'numOfRows': 999, 'type': 'json', **params}, safe='/%')}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            log_list.append(f"네트워크 요청 실패: {e}")
            return []
        except ValueError:
            log_list.append(f"API 응답이 올바른 JSON 형식이 아닙니다: {response.text}")
            return []

        response_data = data.get('response', {})
        header = response_data.get('header', {})
        if header.get('resultCode') != '00':
            log_list.append(f"API 오류: {header.get('resultMsg', '오류 메시지 없음')}")
        return response_data.get('body', {}).get('items', [])

    def get_pre_standard_specs(self, start_date, end_date, log_list):
        params = {'rgstBgnDt': start_date, 'rgstEndDt': end_date}
        return self._make_request("getDataSetOpnStdPrdnmInfo", params, log_list)
    def get_bid_announcements(self, start_date, end_date, log_list):
        params = {'bidNtceBgnDt': start_date, 'bidNtceEndDt': end_date}
        return self._make_request("getDataSetOpnStdBidPblancInfo", params, log_list)
    def get_successful_bid_info(self, start_date, end_date, log_list, bsns_div_cd):
        params = {'opengBgnDt': start_date, 'opengEndDt': end_date, 'bsnsDivCd': bsns_div_cd}
        return self._make_request("getDataSetOpnStdScsbidInfo", params, log_list)
    def get_contract_info(self, start_date, end_date, log_list):
        params = {'cntrctCnclsBgnDate': start_date, 'cntrctCnclsEndDate': end_date}
        return self._make_request("getDataSetOpnStdCntrctInfo", params, log_list)


def search_and_process(api_client, fetch_function, date_params, keywords, search_field, log_list, **kwargs):
    start_date, end_date = date_params
    log_list.append(f"기간({search_field}): {start_date} ~ {end_date} 조회 시작...")
    raw_data = fetch_function(start_date=start_date, end_date=end_date, log_list=log_list, **kwargs)
    if not raw_data: log_list.append("조회된 데이터가 없습니다."); return pd.DataFrame()
    log_list.append(f"총 {len(raw_data)}건 수신. 키워드 필터링 시작...")
    f = [i for i in raw_data if i and search_field in i and i[search_field] and any(k.lower() in i[search_field].lower() for k in keywords)]
    log_list.append(f"필터링 후 {len(f)}건 발견.")
    return pd.DataFrame(f)

# --- 3. 데이터베이스 ---
def setup_database():
    conn = sqlite3.connect("procurement_data.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS projects (bidNtceNo TEXT PRIMARY KEY, bidNtceNm TEXT, ntcelnsttNm TEXT, presmptPrce INTEGER, bid_status TEXT DEFAULT '공고', bidNtceDate TEXT, sucsfCorpNm TEXT, cntrctAmt INTEGER, cntrctDate TEXT)")
    try:
        for col in ['prestandard_status', 'prestandard_no', 'prestandard_date']: cursor.execute(f"ALTER TABLE projects ADD COLUMN {col} TEXT")
    except sqlite3.OperationalError: pass
    conn.commit(); conn.close()

def upsert_project_data(df, stage):
    if df.empty: return
    conn = sqlite3.connect("procurement_data.db")
    for _, r in df.iterrows():
        if stage=='bid': conn.execute("INSERT OR IGNORE INTO projects (bidNtceNo, bidNtceNm, ntcelnsttNm, presmptPrce, bidNtceDate, prestandard_status, prestandard_no, prestandard_date) VALUES (?,?,?,?,?,?,?,?)", (r.get('bidNtceNo'),r.get('bidNtceNm'),r.get('ntcelnsttNm'),r.get('presmptPrce'),r.get('bidNtceDate'),r.get('prestandard_status'),r.get('prestandard_no'),r.get('prestandard_date')))
        elif stage=='successful_bid': conn.execute("UPDATE projects SET bid_status='낙찰', sucsfCorpNm=? WHERE bidNtceNo=? AND bid_status='공고'",(r.get('sucsfCorpNm'),r.get('bidNtceNo')))
        elif stage=='contract': conn.execute("UPDATE projects SET bid_status='계약완료', sucsfCorpNm=?, cntrctAmt=?, cntrctDate=? WHERE bidNtceNo=?",(r.get('rprsntCorpNm'),r.get('cntrctAmt'),r.get('cntrctCnclsDate'),r.get('bidNtceNo')))
    conn.commit(); conn.close()


# --- 4. AI 분석, 리스크 분석 및 보고서 ---
def get_gemini_analysis(api_key, df, log_list):
    if df.empty: log_list.append("AI가 분석할 데이터가 없습니다."); return None
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        log_list.append("Gemini API로 맞춤형 전략 분석 시작...")
        data_for_prompt = df[[c for c in ['prestandard_status', 'bidNtceNm', 'bid_status', 'ntcelnsttNm', 'presmptPrce', 'sucsfCorpNm', 'cntrctAmt'] if c in df.columns]].head(30).to_string()
        prompt = f"""당신은 '위치 및 동작 인식 기술'을 기반으로 VR/MR/AR/XR 가상환경과 현실공간을 정밀하게 매칭(공간 정합)하는 기술을 보유한, 군/경 훈련 시뮬레이터 전문 기업의 '신사업 전략팀장'입니다. 아래 제공된 나라장터 조달 데이터를 바탕으로, 우리 회사의 기술적 강점을 극대화하고 새로운 사업 기회를 발굴하기 위한 심층 분석 보고서를 작성해주세요. 결과는 반드시 아래 형식에 맞춰 마크다운으로 작성해주세요.\n\n[분석 데이터]\n{data_for_prompt}\n\n---\n## 군·경 훈련 시뮬레이션 사업 기회 분석 보고서\n\n### 1. 총평 및 핵심 동향\n(우리 회사의 XR 및 공간 정합 기술과 관련성이 높은 훈련 시뮬레이션, 가상현실, MRO 사업의 증감 추세나, 주목해야 할 발주 기관(육군, 경찰청 등)의 동향을 분석해주세요.)\n\n### 2. 주요 사업 심층 분석\n(우리 회사 기술과의 연관성이 가장 높거나 사업적 가치가 큰 프로젝트 3~5개를 선정하여 아래 표 형식으로 분석해주세요. '분석 및 제언' 항목에는 **우리 회사의 핵심 기술인 '공간 정합', '위치/동작 인식' 기술을 적용할 수 있는 지점이나, 기존 시스템을 고도화할 수 있는 사업 기회**를 중점적으로 작성해주세요.)\n\n| 사업명 | 발주기관 | 추정가격/계약금액 | 진행 상태 | 분석 및 제언 (자사 기술 연계 방안) |\n|---|---|---|---|---|\n| (사업명) | (기관명) | (금액) | (상태) | (예: 이 사업은 CQB 훈련 시뮬레이터로, **현실 공간과 가상 훈련 시나리오를 정밀하게 매칭하는 우리 기술**이 핵심 경쟁력이 될 수 있음.) |\n\n### 3. 기술 연계 가능 키워드\n(데이터에서 식별된 키워드 중, 우리 회사의 '위치/동작 인식' 및 'XR 가시화' 기술과 직접적으로 연결될 수 있는 핵심 키워드를 5개 이상 선정하여 불렛 포인트로 나열해주세요.)\n\n### 4. 차기 사업 전략 제언\n(위 분석을 종합하여, 우리 회사가 다음 분기에 집중해야 할 사업 영역, 기술 고도화 방향 등에 대한 구체적인 실행 전략을 1~2가지 제언해주세요.)"""
        response = model.generate_content(prompt); log_list.append("Gemini 맞춤형 전략 분석 완료.")
        return response.text
    except Exception as e: log_list.append(f"Gemini API 호출 중 오류 발생: {e}"); return None

def expand_keywords_with_gemini(api_key, df, existing_keywords, log_list):
    if df.empty: log_list.append("키워드 확장을 위한 분석 데이터가 없습니다."); return set()
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-flash')
        log_list.append("Gemini API로 지능형 키워드 확장 시작...")
        project_titles = pd.concat([df[col] for col in ['bidNtceNm', 'cntrctNm'] if col in df.columns]).dropna().unique()
        project_titles_str = '\n- '.join(project_titles[:30])
        prompt = f"""당신은 군/경 훈련 시뮬레이터 전문 기업의 조달 정보 분석가입니다. 우리 회사는 '위치/동작 인식', 'XR 공간 정합' 기술을 보유하고 있습니다. 아래는 현재 우리가 사용중인 검색 키워드와, 최근 조달 시스템에서 발견된 사업명 목록입니다. 이 정보를 바탕으로, 우리 회사의 기술과 관련성이 높으면서도 기존 키워드에 없는 **새로운 검색 키워드를 5~10개 추천**해주세요. 결과는 다른 설명 없이, 쉼표(,)로 구분된 키워드 목록으로만 제공해주세요.\n\n[기존 키워드]\n{', '.join(sorted(list(existing_keywords)))}\n\n[최근 발견된 사업명]\n- {project_titles_str}\n\n[추천 키워드]"""
        response = model.generate_content(prompt)
        new_keywords = {k.strip() for k in response.text.strip().split(',') if k.strip()}
        log_list.append(f"Gemini가 추천한 신규 키워드: {new_keywords}")
        return new_keywords
    except Exception as e: log_list.append(f"Gemini 키워드 확장 중 오류 발생: {e}"); return set()

def analyze_project_risk(df: pd.DataFrame) -> pd.DataFrame:
    ongoing_df = df[df['bid_status'].isin(['공고', '낙찰'])].copy()
    if ongoing_df.empty: return pd.DataFrame()
    ongoing_df['score'] = 0; ongoing_df['risk_reason'] = ''
    ongoing_df['bidNtceDate_dt'] = pd.to_datetime(ongoing_df['bidNtceDate'], errors='coerce')
    
    current_time = datetime.now() # 분석 시점 고정

    for index, row in ongoing_df.iterrows():
        reasons = []
        if pd.notna(row['bidNtceDate_dt']) and (current_time - row['bidNtceDate_dt']).days > 30 and row['bid_status'] == '공고':
            ongoing_df.loc[index, 'score'] += 5; reasons.append('공고 후 30일 경과')
        if row.get('prestandard_status') == '해당 없음':
            ongoing_df.loc[index, 'score'] += 3; reasons.append('사전규격 없음')
        try:
            # 금액 데이터는 create_report_data에서 포맷팅되어 쉼표가 포함될 수 있으므로 제거 후 변환
            price = int(str(row.get('presmptPrce', '0')).replace(',', ''))
            if 0 < price < 50000000: ongoing_df.loc[index, 'score'] += 2; reasons.append('소규모 사업')
        except (ValueError, TypeError): pass
        ongoing_df.loc[index, 'risk_reason'] = ', '.join(reasons)
    ongoing_df['risk_level'] = ongoing_df['score'].apply(lambda s: '높음' if s >= 7 else ('보통' if s >= 4 else '낮음'))
    risk_table = ongoing_df[['bidNtceNm', 'ntcelnsttNm', 'bid_status', 'risk_level', 'risk_reason']]
    return risk_table.rename(columns={'bidNtceNm':'사업명','ntcelnsttNm':'발주기관','bid_status':'진행 상태','risk_level':'리스크 등급','risk_reason':'주요 리스크'}).sort_values(by='리스크 등급',key=lambda x:x.map({'높음':0,'보통':1,'낮음':2}))

def create_report_data(db_path, keywords, log_list):
    log_list.append("DB에서 최종 데이터 조회 중...")
    conn = sqlite3.connect(db_path); flat_df = pd.DataFrame()
    try:
        all_projects_df = pd.read_sql_query("SELECT * FROM projects ORDER BY bidNtceDate DESC", conn)
        if all_projects_df.empty: log_list.append("DB에 프로젝트 데이터가 없습니다."); return None
        flat_df = all_projects_df[all_projects_df['bidNtceNm'].str.contains('|'.join(keywords),case=False,na=False)].copy()
        if flat_df.empty: log_list.append("키워드에 해당하는 프로젝트가 없습니다."); return None
        
        # 날짜 포맷팅
        for col in ['prestandard_date','bidNtceDate','cntrctDate']:
            if col in flat_df.columns: flat_df[col] = pd.to_datetime(flat_df[col],errors='coerce').dt.strftime('%Y-%m-%d')
        
        # 금액 포맷팅 (안정성 향상)
        def format_price(x):
            if pd.isna(x):
                return ""
            try:
                # float로 먼저 변환하여 실수형 데이터도 처리 가능하게 함
                return f"{int(float(x)):,}"
            except (ValueError, TypeError):
                # 변환 실패 시 빈 문자열 반환
                return ""

        for col in ['presmptPrce','cntrctAmt']:
            if col in flat_df.columns: 
                flat_df[col] = flat_df[col].apply(format_price)

        structured_columns = {('프로젝트 개요','사업명'):flat_df.get('bidNtceNm'), ('프로젝트 개요','발주기관'):flat_df.get('ntcelnsttNm'), ('진행 현황','종합 상태'):flat_df.get('bid_status'), ('진행 현황','낙찰/계약사'):flat_df.get('sucsfCorpNm'), ('사전 규격 정보','공개 상태'):flat_df.get('prestandard_status'), ('사전 규격 정보','공개일'):flat_df.get('prestandard_date'), ('입찰 공고 정보','공고일'):flat_df.get('bidNtceDate'), ('입찰 공고 정보','추정가격'):flat_df.get('presmptPrce'), ('계약 체결 정보','계약일'):flat_df.get('cntrctDate'), ('계약 체결 정보','계약금액'):flat_df.get('cntrctAmt'), ('참조 번호','사전규격번호'):flat_df.get('prestandard_no'), ('참조 번호','입찰공고번호'):flat_df.get('bidNtceNo')}
        structured_df = pd.DataFrame(structured_columns)
        log_list.append("보고서용 데이터프레임 생성 완료.")
        return {"flat": flat_df, "structured": structured_df}
    except Exception as e: log_list.append(f"보고서 데이터 생성 중 오류: {e}"); return None
    finally: conn.close()


# --- 5. 메인 실행 함수 ---
def run_analysis(search_keywords: set, client: NaraJangteoApiClient, gemini_key: str, auto_expand_keywords: bool = True):
    log = []; all_found_data = {}
    # 분석 시작 시점의 시간으로 고정하여 일관성 유지
    fixed_end_date = datetime.now()
    log.append(f"💡 정보: 현재 날짜 {fixed_end_date.strftime('%Y-%m-%d')} 기준으로 최신 데이터를 검색합니다.")
    start_dt_60d, start_dt_28d, start_dt_6d = fixed_end_date-timedelta(days=60), fixed_end_date-timedelta(days=28), fixed_end_date-timedelta(days=6)
    
    all_found_data['pre_standard'] = search_and_process(client, client.get_pre_standard_specs, (start_dt_60d.strftime('%Y%m%d'), fixed_end_date.strftime('%Y%m%d')), search_keywords, 'prdctClsfcNoNm', log)
    pre_standard_map = {r['bfSpecRgstNo']: r for _, r in all_found_data['pre_standard'].iterrows() if 'bfSpecRgstNo' in r and r['bfSpecRgstNo']} if not all_found_data['pre_standard'].empty else {}
    
    bid_df = search_and_process(client, client.get_bid_announcements, (start_dt_28d.strftime('%Y%m%d%H%M'), fixed_end_date.strftime('%Y%m%d%H%M')), search_keywords, 'bidNtceNm', log)
    if not bid_df.empty:
        def link_pre_standard(row):
            spec_no = row.get('bfSpecRgstNo')
            return ("확인", spec_no, pre_standard_map[spec_no].get('registDt')) if spec_no and spec_no in pre_standard_map else ("해당 없음", None, None)
        bid_df[['prestandard_status', 'prestandard_no', 'prestandard_date']] = bid_df.apply(link_pre_standard, axis=1, result_type='expand')
    all_found_data['bid'] = bid_df
    upsert_project_data(bid_df, 'bid')
    
    succ_dfs = [df for code in ['1','2','3','5'] if not (df := search_and_process(client, client.get_successful_bid_info, (start_dt_6d.strftime('%Y%m%d%H%M'), fixed_end_date.strftime('%Y%m%d%H%M')), search_keywords, 'bidNtceNm', log, bsns_div_cd=code)).empty]
    all_found_data['successful_bid'] = pd.concat(succ_dfs, ignore_index=True) if succ_dfs else pd.DataFrame()
    if not all_found_data['successful_bid'].empty: upsert_project_data(all_found_data['successful_bid'], 'successful_bid')

    all_found_data['contract'] = search_and_process(client, client.get_contract_info, (start_dt_28d.strftime('%Y%m%d'), fixed_end_date.strftime('%Y%m%d')), search_keywords, 'cntrctNm', log)
    if not all_found_data['contract'].empty: upsert_project_data(all_found_data['contract'], 'contract')
    
    report_dfs = create_report_data("procurement_data.db", list(search_keywords), log)
    risk_df, report_data_bytes, gemini_report = pd.DataFrame(), None, None

    if report_dfs:
        risk_df = analyze_project_risk(report_dfs["flat"])
        excel_sheets = {"종합 현황 보고서": report_dfs["structured"], "리스크 분석": risk_df, "사전규격 원본": all_found_data.get('pre_standard'), "입찰공고 원본": all_found_data.get('bid'), "낙찰정보 원본": all_found_data.get('successful_bid'), "계약정보 원본": all_found_data.get('contract')}
        # 수정된 함수 호출로 오류 해결
        report_data_bytes = save_integrated_excel(excel_sheets)

    if gemini_key and report_dfs:
        gemini_report = get_gemini_analysis(gemini_key, report_dfs["flat"], log)
        if auto_expand_keywords and any(not df.empty for df in all_found_data.values()):
            combined_df = pd.concat([df for df in all_found_data.values() if not df.empty], ignore_index=True)
            new_keywords = expand_keywords_with_gemini(gemini_key, combined_df, search_keywords, log)
            if new_keywords:
                updated_keywords = search_keywords.union(new_keywords)
                save_keywords(updated_keywords)
                log.append("키워드 파일이 새롭게 확장되었습니다!")
    
    # 파일 이름도 고정된 시간을 사용
    return {"log": log, "risk_df": risk_df, "report_file_data": report_data_bytes, "report_filename": f"integrated_report_{fixed_end_date.strftime('%Y%m%d_%H%M%S')}.xlsx", "gemini_report": gemini_report}