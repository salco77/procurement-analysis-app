"""
AI 기반 국방/경찰 조달 정보 분석 시스템
Version: 14.2 - 계약현황 API 수정 버전
"""

import os
import requests
import requests.adapters
import urllib3
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta 
import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import xmltodict
from typing import Optional, Dict, List, Any, Tuple, Set
import sqlite3
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Gemini AI SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai library not found. AI analysis will be disabled.")

# ============================================
# 0. 유틸리티 함수
# ============================================

def standardize_date(date_str: Any) -> Optional[str]:
    """다양한 형식의 날짜 문자열을 표준 ISO 형식(YYYY-MM-DD HH:MM:SS)으로 변환합니다."""
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    if not date_str:
        return None
    
    # 공공 API에서 자주 사용되는 형식 정의
    formats = [
        '%Y%m%d%H%M%S', # 예: 20250828143000
        '%Y%m%d %H%M',  # 예: 20250828 1430
        '%Y%m%d%H%M',   # 예: 202508281430
        '%Y%m%d',       # 예: 20250828
        '%Y-%m-%d %H:%M:%S', # ISO 표준
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    
    # pandas를 이용한 추가적인 파싱 시도 (더 유연함)
    try:
            dt = pd.to_datetime(date_str, errors='coerce')
            if not pd.isna(dt):
                return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
            pass

    # logger.warning(f"Could not parse date format: {date_str}")
    return None

# ============================================
# 1. 회사 프로필 및 키워드 설정
# ============================================
COMPANY_PROFILE = """
우리 회사는 국방/경찰 분야의 종합 솔루션 제공 기업으로 다음 사업을 수행합니다:
[1. VR/AR/XR 훈련 시뮬레이션 사업]
- 영상 모의 사격 훈련 시스템 및 과학화 사격장, 비행/항공/헬기 시뮬레이터, 전차/장갑차 운용 시뮬레이터 개발 및 납품 경험 다수 보유.
[2. 무기체계 정비/유지보수 사업 (MRO)]
- 유도무기 체계(비궁, 천무 등) 및 화력장비(K9, K2) 정비 지원 시스템 개발 및 MRO 수행.
[3. 첨단기술 국방 솔루션]
- AI 기반 훈련평가 및 예측정비, IoT/빅데이터 기반 장비 관리 시스템 연구개발.
"""

# 키워드 정의 및 가중치 설정
SIMULATION_CORE_KEYWORDS = {"시뮬레이터", "모의훈련", "VR", "XR", "과학화사격장", "영상사격", "가상현실"}
WEAPON_CORE_KEYWORDS = {"비궁", "천무", "유도무기", "K9", "K2", "현무"}
MRO_KEYWORDS = {"정비", "유지보수", "MRO", "PBL", "성능개량", "창정비"}
ADVANCED_TECH_KEYWORDS = {"AI", "빅데이터", "드론", "IoT", "예측정비", "인공지능"}
GENERAL_KEYWORDS = {"시스템", "체계", "개발", "연구용역", "솔루션"}
BROAD_KEYWORDS = {"국방", "경찰", "연구", "훈련", "사업"}
NEGATIVE_KEYWORDS = {"가상계좌", "모의고사", "토목", "건축", "급식", "사무용품", "인쇄"}

KEYWORD_WEIGHTS = {
    "핵심_시뮬레이션": 15, "핵심_무기체계": 15, "MRO": 10, "첨단기술": 8, "일반": 5, "광범위": 1
}

# ============================================
# 2. 데이터 관리 및 분석 함수 (DataManager 및 벡터화 엔진)
# ============================================

class DataManager:
    """SQLite 기반 데이터 관리자 (성능 최적화 적용)"""
    def __init__(self, db_path='procurement_data.db'):
        self.db_path = db_path
        logger.info(f"Initializing DataManager. DB Path: {os.path.abspath(self.db_path)}")
        self._initialize_db()

    def _get_connection(self):
        """DB 연결을 생성하고 성능 최적화 PRAGMA를 적용합니다."""
        conn = sqlite3.connect(self.db_path)
        try:
            # 성능 개선: SQLite 튜닝 (WAL 모드)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except sqlite3.DatabaseError as e:
            logger.warning(f"Failed to set SQLite PRAGMA settings: {e}")
        return conn

    def _initialize_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 테이블 및 인덱스 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS raw_data (
                        UID TEXT PRIMARY KEY, ID TEXT, Title TEXT, Source TEXT, Type TEXT, Agency TEXT, Budget REAL,
                        AnnouncementDate TEXT, Deadline TEXT, OpeningDate TEXT, ContractDate TEXT,
                        Details TEXT, Link TEXT, Winner TEXT, CollectedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        UID TEXT PRIMARY KEY, Score INTEGER, MatchedKeywords TEXT, AI_Analysis TEXT, AI_Strategy TEXT,
                        AnalyzedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (UID) REFERENCES raw_data (UID)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_announcement_date ON raw_data (AnnouncementDate);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_contract_date ON raw_data (ContractDate);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_score ON analysis_results (Score);")
                conn.commit()
            logger.info("Database initialized and optimized (WAL mode enabled).")
        except Exception as e:
            logger.error(f"CRITICAL: Database initialization failed: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")

    @staticmethod
    def _generate_uid(item: Dict[str, Any]):
        identifier = item.get('ID') or f"{item.get('Title')}_{item.get('Agency')}"
        key_string = f"{item.get('Source')}_{item.get('Type')}_{identifier}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    # 성능 개선: 데이터 저장 및 신규 데이터 식별 함수
    def save_and_identify_new_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """데이터를 저장(UPSERT)하고, 완전히 새로운 데이터(UID 기준)를 식별하여 반환합니다."""
        if df.empty: return pd.DataFrame(), []
        
        df_save = df.copy()
        df_save['UID'] = df_save.apply(self._generate_uid, axis=1)
        collected_uids = df_save['UID'].tolist()
        
        # 1. 신규 데이터 식별
        new_uids = []
        try:
            with self._get_connection() as conn:
                # DB에 이미 존재하는 UID 조회 (대량 조회 최적화)
                query = f"SELECT UID FROM raw_data WHERE UID IN ({','.join(['?']*len(collected_uids))})"
                existing_uids_df = pd.read_sql_query(query, conn, params=collected_uids)
                existing_uids = set(existing_uids_df['UID'])
                
                new_uids = [uid for uid in collected_uids if uid not in existing_uids]

            logger.info(f"Collected {len(df_save)} items. Identified {len(new_uids)} as NEW data for analysis.")

            # 2. 데이터 저장 (UPSERT)
            cols_to_save = ['UID', 'ID', 'Title', 'Source', 'Type', 'Agency', 'Budget', 
                            'AnnouncementDate', 'Deadline', 'OpeningDate', 'ContractDate',
                            'Details', 'Link', 'Winner']
            for col in cols_to_save:
                if col not in df_save.columns:
                    df_save[col] = None
            df_save = df_save.where(pd.notnull(df_save), None)

            with self._get_connection() as conn:
                # 임시 테이블을 사용한 효율적인 UPSERT
                df_save[cols_to_save].to_sql('temp_raw_data', conn, if_exists='replace', index=False)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO raw_data (UID, ID, Title, Source, Type, Agency, Budget, AnnouncementDate, Deadline, OpeningDate, ContractDate, Details, Link, Winner, CollectedAt)
                    SELECT UID, ID, Title, Source, Type, Agency, Budget, AnnouncementDate, Deadline, OpeningDate, ContractDate, Details, Link, Winner, CURRENT_TIMESTAMP
                    FROM temp_raw_data
                """)
                cursor.execute("DROP TABLE temp_raw_data")
                conn.commit()
            logger.info(f"Saved/Updated {len(df_save)} raw data entries to DB.")

        except Exception as e:
                logger.error(f"DB 저장 또는 신규 데이터 식별 중 오류 발생: {e}")
                return df_save, collected_uids # 오류 시 전체를 신규로 간주

        return df_save, new_uids

    def save_analysis_results(self, df: pd.DataFrame):
        """분석 결과를 DB에 저장합니다."""
        if df.empty or 'UID' not in df.columns: return 0
        cols_to_save = ['UID', 'Score', 'MatchedKeywords', 'AI_Analysis', 'AI_Strategy']
        df_save = df.copy()
        for col in cols_to_save:
            if col not in df_save.columns: df_save[col] = None
        df_save = df_save.where(pd.notnull(df_save), None)

        try:
            with self._get_connection() as conn:
                df_save[cols_to_save].to_sql('temp_analysis_results', conn, if_exists='replace', index=False)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_results (UID, Score, MatchedKeywords, AI_Analysis, AI_Strategy, AnalyzedAt)
                    SELECT UID, Score, MatchedKeywords, AI_Analysis, AI_Strategy, CURRENT_TIMESTAMP
                    FROM temp_analysis_results
                """)
                cursor.execute("DROP TABLE temp_analysis_results")
                conn.commit()
        except Exception as e:
            logger.error(f"DB 분석 결과 저장 중 오류 발생: {e}")
        logger.info(f"Saved/Updated {len(df_save)} analysis results to DB.")
        return len(df_save)

    def load_data(self, start_date: date, end_date: date, min_score: int):
        # 날짜 형식은 YYYY-MM-DD로 통일하여 쿼리 (DB에 표준화되어 저장되므로)
        start_str = start_date.strftime('%Y-%m-%d')
        # 종료일 다음 날의 00:00 이전까지 조회하여 종료일 포함
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        query = """
            SELECT R.*, A.Score, A.MatchedKeywords, A.AI_Analysis, A.AI_Strategy
            FROM raw_data R
            JOIN analysis_results A ON R.UID = A.UID
            WHERE (
                (R.AnnouncementDate >= ? AND R.AnnouncementDate < ?) OR
                (R.ContractDate >= ? AND R.ContractDate < ?)
            )
            AND A.Score >= ?
        """
        params = (start_str, end_str, start_str, end_str, min_score)

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Failed to load data from DB: {e}")
            return pd.DataFrame()
        return df

    def get_last_collection_date(self):
        query = "SELECT MAX(CollectedAt) FROM raw_data"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchone()
            return result[0] if result and result[0] else None
        except Exception:
            return None

# --- 고성능 벡터화 분석 엔진 ---

def load_keywords():
    return {
        "핵심_시뮬레이션": SIMULATION_CORE_KEYWORDS, "핵심_무기체계": WEAPON_CORE_KEYWORDS,
        "MRO": MRO_KEYWORDS, "첨단기술": ADVANCED_TECH_KEYWORDS, "일반": GENERAL_KEYWORDS,
        "광범위": BROAD_KEYWORDS, "제외": NEGATIVE_KEYWORDS
    }

def compile_keyword_patterns(keywords):
    patterns = {}
    for category, kws in keywords.items():
        if kws:
            # 키워드를 '|'로 연결하여 하나의 정규 표현식으로 만듦 (대소문자 무시)
            pattern = '({})'.format('|'.join(re.escape(kw) for kw in kws))
            patterns[category] = re.compile(pattern, re.IGNORECASE)
    return patterns

def analyze_data_vectorized(df, keywords, min_score):
    """벡터화 연산을 사용하여 데이터프레임 전체를 고속으로 분석합니다."""
    if df.empty:
        return pd.DataFrame()

    logger.info(f"Starting keyword analysis on {len(df)} items (Vectorized Engine)...")
    df_analyzed = df.copy()
    df_analyzed['Text_Analyze'] = df_analyzed['Title'].fillna('') + '\n' + df_analyzed['Details'].fillna('')

    # 1. 패턴 컴파일
    patterns = compile_keyword_patterns(keywords)

    # 2. 제외 키워드 처리 (벡터화)
    if '제외' in patterns:
        mask_exclude = df_analyzed['Text_Analyze'].str.contains(patterns['제외'], regex=True, na=False)
        df_analyzed.loc[mask_exclude, 'Score'] = -100
        df_analyzed.loc[mask_exclude, 'MatchedKeywords'] = df_analyzed.loc[mask_exclude, 'Text_Analyze'].str.findall(patterns['제외']).apply(lambda x: f"제외: {', '.join(set(map(str.lower, x)))}")
    else:
        mask_exclude = pd.Series([False] * len(df_analyzed), index=df_analyzed.index)

    # 분석 대상 데이터 필터링
    df_target = df_analyzed[~mask_exclude].copy()
    if df_target.empty:
        return df_analyzed.drop(columns=['Text_Analyze'], errors='ignore')

    # 3. 키워드 매칭 및 점수 계산 (벡터화)
    scores = pd.Series(0, index=df_target.index)
    all_matches = {}

    for category, pattern in patterns.items():
        if category == '제외':
            continue
        
        weight = KEYWORD_WEIGHTS.get(category, 1)
        matches = df_target['Text_Analyze'].str.findall(pattern)
        
        if category == '광범위':
            scores += matches.apply(lambda x: weight if x else 0)
        else:
            # 고유 매칭 키워드 수 계산 (소문자로 통일하여 중복 제거)
            match_counts = matches.apply(lambda x: len(set(map(str.lower, x))))
            scores += match_counts * weight
            
            for idx, match_list in matches.items():
                if match_list:
                    if idx not in all_matches:
                        all_matches[idx] = set()
                    all_matches[idx].update(set(map(str.lower, match_list)))

    # 4. 결과 병합
    df_target['Score'] = scores
    df_target['MatchedKeywords'] = pd.Series(all_matches).apply(lambda x: ", ".join(sorted(list(x))) if isinstance(x, set) else "")
    
    df_final = pd.concat([df_analyzed[mask_exclude], df_target])
    df_final = df_final.sort_values(by='Score', ascending=False)

    logger.info("Keyword analysis complete.")
    return df_final.drop(columns=['Text_Analyze'], errors='ignore')


# ============================================
# 3. AI 분석 엔진 (Gemini 연동)
# ============================================

class GeminiAnalyzer:
    """Gemini AI를 사용하여 조달 정보를 심층 분석합니다."""
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            self.model = None; return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini Analyzer initialized successfully.")
        except Exception as e:
            logger.error(f"Gemini AI 초기화 실패: {e}"); self.model = None

    def generate_prompt(self, item: Dict[str, Any]):
        details_short = str(item.get('Details', ''))[:1500]
        budget_val = item.get('Budget')
        budget_str = f"{float(budget_val):,.0f} 원" if budget_val is not None and not pd.isna(budget_val) and budget_val > 0 else "정보 없음"

        prompt = f"""
        당신은 국방 조달 전문 분석가입니다. '회사 프로필' 기반으로 '조달 정보'의 적합성을 분석하고 전략을 제안하세요.

        [회사 프로필]
        {COMPANY_PROFILE}

        [조달 정보]
        유형: {item.get('Type')}, 제목: {item.get('Title')}, 기관: {item.get('Agency')}, 예산/금액: {budget_str}
        공고일(예정일): {item.get('AnnouncementDate', 'N/A')}, 마감일/개찰일: {item.get('Deadline') or item.get('OpeningDate', 'N/A')}
        상세 내용: {details_short}...

        [분석 지침]
        1. 핵심 역량과의 연관성 분석. 2. 강점과 기회 언급. 3. 구체적인 참여 전략 제안.
        4. 응답은 반드시 아래 JSON 형식으로 제공. 다른 설명 생략.

        [응답 형식 (JSON)]
        {{
            "analysis_summary": "...",
            "strategy_proposal": "..."
        }}
        """
        return prompt

    def analyze_item(self, item: Dict[str, Any]):
        if not self.model: return {"AI_Analysis": "AI 엔진 미작동", "AI_Strategy": "N/A"}
        prompt = self.generate_prompt(item); response_text = ""
        try:
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.3))
            response_text = response.text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"): response_text = response_text[3:-3].strip()
            
            result = json.loads(response_text)
            return {"AI_Analysis": result.get("analysis_summary", "분석 실패"), "AI_Strategy": result.get("strategy_proposal", "제안 실패")}
        except Exception as e:
            logger.error(f"AI 분석 중 오류 발생: {item.get('ID')} - {e}")
            return {"AI_Analysis": f"분석 오류: {e}", "AI_Strategy": "N/A"}

    def run_batch_analysis(self, df: pd.DataFrame):
        """데이터프레임 전체를 병렬로 분석합니다."""
        if not self.model or df.empty:
            df_result = df.copy()
            df_result['AI_Analysis'] = "N/A (AI 미실행)"; df_result['AI_Strategy'] = "N/A"
            return df_result

        logger.info(f"Starting AI deep dive analysis on {len(df)} items (Parallel execution)...")
        results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(self.analyze_item, row.to_dict()): index for index, row in df.iterrows()}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results.append({'index': index, **future.result()})
                except Exception as e:
                    results.append({'index': index, "AI_Analysis": "처리 중 오류", "AI_Strategy": "N/A"})

        df_ai = pd.DataFrame(results).set_index('index')
        df_merged = df.join(df_ai, how='left')
        logger.info("AI analysis complete.")
        return df_merged


# ============================================
# 4. API 클라이언트 정의 (계약현황 수정됨)
# ============================================

class BaseApiClient:
    """API 호출을 위한 기본 클라이언트 (세션 관리 및 재시도 로직 포함)"""
    def __init__(self, api_key: str):
        self.api_key = api_key; self.session = self._create_session()
    
    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter); session.mount('https://', adapter)
        return session

    def _call_api(self, url: str, params: Dict[str, Any], response_format: str = 'json'):
        if not self.api_key: return None
        if 'ServiceKey' not in params: params['ServiceKey'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30, verify=False)
            response.raise_for_status()
            if response_format == 'json': return response.json()
            elif response_format == 'xml':
                response.encoding = 'utf-8'; return xmltodict.parse(response.text)
        except Exception as e:
            logger.error(f"API 호출 실패: {url} - Error: {e}"); return None

class NaraJangteoApiClient(BaseApiClient):
    """나라장터(g2b) API 클라이언트 (고성능 병렬 처리 및 날짜 표준화)"""
    BASE_URL = "http://apis.data.go.kr/1230000"
    PAGE_SIZE = 500

    def _generate_monthly_ranges(self, start_date, end_date):
        ranges = []
        current_start = start_date
        while current_start <= end_date:
            month_end = current_start + relativedelta(day=31)
            current_end = min(month_end, end_date)
            ranges.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
        return ranges

    def _fetch_parallel_monthly_chunks(self, endpoint, start_date, end_date, date_param_prefix=None, api_type='standard'):
        """월 단위로 분할된 기간에 대해 병렬로 데이터를 가져옵니다 (성능 핵심)."""
        date_ranges = self._generate_monthly_ranges(start_date, end_date)
        all_items = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for chunk_start, chunk_end in date_ranges:
                # API 타입에 따른 날짜 형식 및 파라미터 설정
                if api_type == 'contract':
                    # 계약현황 API 전용 처리
                    start_str = chunk_start.strftime('%Y%m%d')
                    end_str = chunk_end.strftime('%Y%m%d')
                    params_base = {
                        'cntrctBgnDate': start_str,
                        'cntrctEndDate': end_str
                    }
                else:
                    # 일반 API (발주계획, 사전규격, 입찰공고)
                    start_str = chunk_start.strftime('%Y%m%d0000')
                    end_str = chunk_end.strftime('%Y%m%d2359')
                    params_base = {
                        f'{date_param_prefix}BgnDt': start_str,
                        f'{date_param_prefix}EndDt': end_str
                    }

                futures.append(executor.submit(self._fetch_paginated_data, endpoint, params_base, None))
            
            for future in as_completed(futures):
                try:
                    all_items.extend(future.result())
                except Exception as e:
                    logger.error(f"NaraJangteo parallel fetch error ({endpoint}): {e}")
        
        return all_items

    def _fetch_paginated_data(self, endpoint, params_base, type_name=None):
        """단일 엔드포인트에서 페이지네이션을 처리하여 모든 데이터를 가져옵니다."""
        all_data = []
        page_no = 1
        max_pages = 10  # 무한 루프 방지
        
        while page_no <= max_pages:
            params = params_base.copy()
            params.update({'pageNo': page_no, 'numOfRows': self.PAGE_SIZE, 'type': 'json'})
            
            # 디버깅을 위한 로깅 추가
            if page_no == 1:
                logger.debug(f"API Call - Endpoint: {endpoint}, Params: {params}")
            
            data = self._call_api(self.BASE_URL + endpoint, params, 'json')

            if data and data.get('response') and data['response'].get('header', {}).get('resultCode') == '00':
                body = data['response']['body']
                
                # items 추출 (다양한 구조 대응)
                items = body.get('items')
                if isinstance(items, dict) and 'item' in items:
                     items = items.get('item')
                if items is None:
                    items = body.get('item')

                if not items: 
                    break

                if not isinstance(items, list):
                    items = [items]

                for item in items:
                    item['_type_name'] = type_name or item.get('bsnsDivNm')
                    all_data.append(item)

                total_count = int(body.get('totalCount', 0))
                if page_no * self.PAGE_SIZE >= total_count: 
                    break
                page_no += 1
            else:
                if page_no == 1:
                    logger.warning(f"No data or error for endpoint {endpoint} with params {params_base}")
                break
                
        return all_data

    # --- 1. 발주계획 ---
    def get_order_plans(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Order Plans...")
        start_ym = start_date.strftime('%Y%m'); end_ym = end_date.strftime('%Y%m')
        endpoints = {
            "물품": "/ao/OrderPlanSttusService/getOrderPlanSttusListThng", 
            "공사": "/ao/OrderPlanSttusService/getOrderPlanSttusListCnstwk", 
            "용역": "/ao/OrderPlanSttusService/getOrderPlanSttusListServc"
        }
        all_items = []
        params_base = {'inqryDiv': '1', 'orderBgnYm': start_ym, 'orderEndYm': end_ym}

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._fetch_paginated_data, endpoint, params_base, type_name): type_name for type_name, endpoint in endpoints.items()}
            for future in as_completed(futures):
                try:
                    all_items.extend(future.result())
                except Exception as e:
                    logger.error(f"Error fetching order plans: {e}")

        normalized_data = []
        for item in all_items:
            details = (f"발주월: {item.get('orderMnth')}, 계약방법: {item.get('cntrctMthdNm')}\n"
                       f"세부품명/규격: {item.get('dtilPrdctClsfcNoNm')} / {item.get('specCntnts')}")
            
            # 날짜 표준화 적용
            announcement_date = standardize_date(item.get('nticeDt'))
            if not announcement_date and item.get('orderYear') and item.get('orderMnth'):
                 try:
                    date_raw = f"{int(item.get('orderYear'))}-{int(item.get('orderMnth')):02d}-01"
                    announcement_date = standardize_date(date_raw)
                 except (ValueError, TypeError):
                    announcement_date = None

            normalized_item = {
                'ID': item.get('orderPlanUntyNo'), 'Title': item.get('bizNm'),
                'AnnouncementDate': announcement_date,
                'Budget': float(item.get('sumOrderAmt') or 0), 'Type': f"발주계획 ({item.get('_type_name')})",
                'Source': '나라장터', 'Agency': item.get('orderInsttNm'),
                'Details': details, 'Link': None, 'Winner': None,
                'Deadline': None, 'OpeningDate': None, 'ContractDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 2. 사전규격 ---
    def get_prior_standards(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Prior Standards...")
        start_dt_str = start_date.strftime('%Y%m%d0000'); end_dt_str = end_date.strftime('%Y%m%d2359')
        endpoints = {
            "물품": "/ao/HrcspSsstndrdInfoService/getPublicPrcureThngInfoThng", 
            "용역": "/ao/HrcspSsstndrdInfoService/getPublicPrcureThngInfoServc", 
            "공사": "/ao/HrcspSsstndrdInfoService/getPublicPrcureThngInfoCnstwk"
        }
        all_items = []
        params_base = {'inqryDiv': '1', 'inqryBgnDt': start_dt_str, 'inqryEndDt': end_dt_str}

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._fetch_paginated_data, endpoint, params_base, type_name): type_name for type_name, endpoint in endpoints.items()}
            for future in as_completed(futures):
                try:
                    all_items.extend(future.result())
                except Exception as e:
                    logger.error(f"Error fetching prior standards: {e}")

        normalized_data = []
        for item in all_items:
            # 날짜 표준화 적용
            deadline = standardize_date(item.get('opninRgstClseDt'))
            announcement_date = standardize_date(item.get('rgstDt'))

            details = (f"담당자: {item.get('ofclNm')}, 마감일: {deadline}\n"
                       f"품목상세: {item.get('prdctDtlList')}")
            agency = item.get('rlDminsttNm') or item.get('orderInsttNm')

            normalized_item = {
                'ID': item.get('bfSpecRgstNo'), 'Title': item.get('prdctClsfcNoNm'), 
                'AnnouncementDate': announcement_date,
                'Deadline': deadline,
                'Budget': float(item.get('asignBdgtAmt') or 0), 'Type': f"사전규격 ({item.get('_type_name')})",
                'Source': '나라장터', 'Agency': agency,
                'Details': details, 'Link': item.get('specDocFileUrl1'),
                'Winner': None, 'OpeningDate': None, 'ContractDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 3. 입찰공고 ---
    def get_bid_announcements(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Bid Announcements (Parallel Monthly chunks)...")
        endpoint = "/ao/PubDataOpnStdService/getDataSetOpnStdBidPblancInfo"
        
        all_items = self._fetch_parallel_monthly_chunks(endpoint, start_date, end_date, 'bidNtce', api_type='standard')

        normalized_data = []
        for item in all_items:
            budget = float(item.get('asignBdgtAmt') or item.get('estmtPrice') or 0)
            
            # 날짜 표준화 적용 (날짜와 시간이 분리된 경우 조합 후 표준화)
            opening_date_raw = f"{item.get('opengDate', '')} {item.get('opengTime', '')}".strip()
            opening_date = standardize_date(opening_date_raw)

            announcement_date_raw = f"{item.get('bidNtceDate', '')} {item.get('bidNtceTime', '')}".strip()
            announcement_date = standardize_date(announcement_date_raw)

            deadline_raw = f"{item.get('bidClseDate', '')} {item.get('bidClseTime', '')}".strip()
            deadline = standardize_date(deadline_raw)

            
            details = (f"업무구분: {item.get('bsnsDivNm')}, 계약방법: {item.get('cntrctMthdNm')}, 공고상태: {item.get('bidNtceSttusNm')}\n"
                       f"입찰마감: {deadline}, 개찰일시: {opening_date}")
            agency = item.get('dminsttNm') or item.get('orderInsttNm') or item.get('ntceInsttNm')

            normalized_item = {
                'ID': f"{item.get('bidNtceNo')}-{item.get('bidNtceOrd')}", 'Title': item.get('bidNtceNm'),
                'AnnouncementDate': announcement_date,
                'OpeningDate': opening_date,
                'Deadline': deadline,
                'Budget': budget,
                'Type': f"입찰공고 ({item.get('_type_name', '기타')})", 'Source': '나라장터', 'Agency': agency,
                'Details': details, 'Link': item.get('bidNtceUrl'), 'Winner': None,
                'ContractDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 4. 계약현황 (수정됨) ---
    def get_contracts(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Contracts (Fixed API)...")
        endpoint = "/ao/PubDataOpnStdService/getDataSetOpnStdCntrctInfo"
        
        # 계약현황 API 전용 처리
        all_items = self._fetch_parallel_monthly_chunks(endpoint, start_date, end_date, None, api_type='contract')
        
        logger.info(f"Total contract items fetched: {len(all_items)}")

        normalized_data = []
        for item in all_items:
            budget = float(item.get('totCntrctAmt') or item.get('cntrctAmt') or 0)
            details = (f"업무구분: {item.get('bsnsDivNm')}, 계약방법: {item.get('cntrctMthdNm')}\n"
                       f"계약기간: {item.get('cntrctBeginDate')} ~ {item.get('cntrctEndDate')}")
            agency = item.get('dminsttNm') or item.get('orderInsttNm')
            winner = item.get('cntrctEntrpsNm') or item.get('sppirdNm')
            
            # 날짜 표준화 적용
            contract_date = standardize_date(item.get('cntrctDate'))

            normalized_item = {
                'ID': item.get('cntrctNo'), 'Title': item.get('cntrctNm'), 
                'ContractDate': contract_date,
                'Budget': budget, 'Type': f"계약현황 ({item.get('_type_name', '기타')})",
                'Source': '나라장터', 'Agency': agency, 'Details': details,
                'Link': item.get('cntrctUrl'), 'Winner': winner,
                'AnnouncementDate': None, 'Deadline': None, 'OpeningDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)


class DapaApiClient(BaseApiClient):
    """방위사업청(DAPA) API 클라이언트 (정식 문서 기반으로 전면 수정)"""
    BASE_URL = "http://openapi.d2b.go.kr/openapi/service" # 1. 올바른 서버 주소로 변경
    PAGE_SIZE = 500

    def _fetch_paginated_data_xml(self, endpoint, params_base, item_key='item', type_name=None):
        """DAPA XML 응답 페이지네이션 처리"""
        all_data = []
        page_no = 1
        while True:
            params = params_base.copy()
            params.update({'pageNo': page_no, 'numOfRows': self.PAGE_SIZE})
            data = self._call_api(self.BASE_URL + endpoint, params, 'xml')

            if data and data.get('response') and data['response'].get('header', {}).get('resultCode') == '00':
                body = data['response'].get('body')
                if not body: break

                items_container = body.get('items')
                if not items_container: break
                
                item_list = items_container.get(item_key)
                if not item_list: break

                if not isinstance(item_list, list):
                    item_list = [item_list]
                
                for item in item_list:
                    if type_name: item['_type_name'] = type_name
                    all_data.append(item)

                total_count = int(body.get('totalCount', 0))
                if page_no * self.PAGE_SIZE >= total_count: break
                page_no += 1
            else:
                error_msg = data['response']['header']['resultMsg'] if data and data.get('response') else "No response"
                logger.warning(f"DAPA API 호출 실패 또는 데이터 없음: {endpoint}, 메시지: {error_msg}")
                break
        return all_data

    # --- 1. 조달계획 (문서 기반으로 재작성) ---
    def get_order_plans(self, start_date, end_date):
        logger.info("Fetching DAPA Order Plans...")
        # API는 월 단위로만 조회 가능
        start_str = start_date.strftime('%Y%m')
        end_str = end_date.strftime('%Y%m')
        params_base = {'orderPrearngeMtBegin': start_str, 'orderPrearngeMtEnd': end_str}
        
        # 국내, 국외, 시설 조달계획 병렬 호출
        plan_items = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._fetch_paginated_data_xml, "/PrcurePlanInfoService/getDmstcPrcurePlanList", params_base, type_name="국내 조달계획"),
                executor.submit(self._fetch_paginated_data_xml, "/PrcurePlanInfoService/getOutnatnPrcurePlanList", {'demandYear': str(start_date.year)}, type_name="국외 조달계획"),
                executor.submit(self._fetch_paginated_data_xml, "/PrcurePlanInfoService/getFcltyPrcurePlanList", params_base, type_name="시설 조달계획")
            ]
            for future in as_completed(futures):
                plan_items.extend(future.result())

        normalized_data = []
        for item in plan_items:
            normalized_item = {
                'ID': item.get('dcsNo') or item.get('purchsRequstNo') or item.get('cntrwkNo'),
                'Title': item.get('reprsntPrdlstNm') or item.get('prdlstNm') or item.get('cntrwkNm'),
                'AnnouncementDate': standardize_date(item.get('orderPrearngeMt', '') + '01'), # 월단위라 1일로 표준화
                'Budget': float(item.get('budgetAmount') or 0),
                'Type': f"{item['_type_name']} (방위사업청)",
                'Source': '방위사업청',
                'Agency': item.get('ornt', '방위사업청'),
                'Details': f"계약방법: {item.get('cntrctMth')}, 진행상태: {item.get('progrsSttus')}",
                'Link': None, 'Winner': None, 'Deadline': None, 'OpeningDate': None, 'ContractDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 2. 입찰공고 (문서 기반으로 재작성) ---
    def get_bid_announcements(self, start_date, end_date):
        logger.info("Fetching DAPA Bid Announcements...")
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        params_base = {'opengDateBegin': start_str, 'opengDateEnd': end_str}

        bid_items = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._fetch_paginated_data_xml, "/BidPblancInfoService/getDmstcCmpetBidPblancList", params_base, type_name="국내 입찰공고"),
                executor.submit(self._fetch_paginated_data_xml, "/BidPblancInfoService/getOutnatnCmpetBidPblancList", params_base, type_name="국외 입찰공고"),
                executor.submit(self._fetch_paginated_data_xml, "/BidPblancInfoService/getFcltyCmpetBidPblancList", params_base, type_name="시설 입찰공고")
            ]
            for future in as_completed(futures):
                bid_items.extend(future.result())

        normalized_data = []
        for item in bid_items:
            normalized_item = {
                'ID': f"{item.get('pblancNo')}-{item.get('pblancOdr')}",
                'Title': item.get('bidNm') or item.get('bsnsNm') or item.get('cntrwkNm'),
                'AnnouncementDate': standardize_date(item.get('pblancDate')),
                'Deadline': standardize_date(item.get('biddocPresentnClosDt')),
                'OpeningDate': standardize_date(item.get('opengDt')),
                'Budget': float(item.get('budgetAmount') or item.get('totAmount') or 0),
                'Type': f"{item['_type_name']} (방위사업청)",
                'Source': '방위사업청',
                'Agency': item.get('ornt', '방위사업청'),
                'Details': f"계약방법: {item.get('cntrctMth')}",
                'Link': None, 'Winner': None, 'ContractDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 3. 계약현황 (문서 기반으로 재작성) ---
    def get_contracts(self, start_date, end_date):
        logger.info("Fetching DAPA Contracts...")
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        params_base = {'cntrctDateBegin': start_str, 'cntrctDateEnd': end_str}
        
        contract_items = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._fetch_paginated_data_xml, "/CntrctInfoService/getDmstcCntrctInfoList", params_base, type_name="국내 계약"),
                executor.submit(self._fetch_paginated_data_xml, "/CntrctInfoService/getOutnatnCntrctInfoList", params_base, type_name="국외 계약"),
                executor.submit(self._fetch_paginated_data_xml, "/CntrctInfoService/getFcltyCntrctInfoList", params_base, type_name="시설 계약")
            ]
            for future in as_completed(futures):
                contract_items.extend(future.result())
        
        normalized_data = []
        for item in contract_items:
            normalized_item = {
                'ID': item.get('cntrctNo'),
                'Title': item.get('cntrctNm'),
                'ContractDate': standardize_date(item.get('cntrctDate')),
                'Budget': float(item.get('cntrctAmnt') or 0),
                'Type': f"계약현황 ({item['_type_name']})",
                'Source': '방위사업청',
                'Agency': item.get('ornt', '방위사업청'),
                'Winner': item.get('cntrctEntrpsNm'),
                'Details': f"계약상태: {item.get('cntrctSttus')}, 계약방법: {item.get('cntrctMth')}",
                'Link': None, 'AnnouncementDate': None, 'Deadline': None, 'OpeningDate': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)


# ============================================
# 5. 메인 분석 실행 함수 (파이프라인 최적화 적용)
# ============================================

def collect_data(nara_client: Optional[NaraJangteoApiClient], 
                 dapa_client: Optional[DapaApiClient], 
                 start_date: date, end_date: date):
    """데이터 수집 프로세스 (병렬 처리 적용)"""
    logger.info(f"Starting data collection from {start_date} to {end_date} (Parallel execution)")

    tasks = []
    # 전체 API 호출 병렬화
    with ThreadPoolExecutor(max_workers=10) as executor:
        if nara_client:
            tasks.append(executor.submit(nara_client.get_order_plans, start_date, end_date))
            tasks.append(executor.submit(nara_client.get_prior_standards, start_date, end_date))
            tasks.append(executor.submit(nara_client.get_bid_announcements, start_date, end_date))
            tasks.append(executor.submit(nara_client.get_contracts, start_date, end_date))

        if dapa_client:
            tasks.append(executor.submit(dapa_client.get_order_plans, start_date, end_date))
            tasks.append(executor.submit(dapa_client.get_bid_announcements, start_date, end_date))
            tasks.append(executor.submit(dapa_client.get_contracts, start_date, end_date))

        all_data_frames = []
        for future in as_completed(tasks):
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_data_frames.append(df)
            except Exception as e:
                logger.error(f"Error during parallel data collection task: {e}", exc_info=True)

    if not all_data_frames:
        return pd.DataFrame()

    df_combined = pd.concat(all_data_frames, ignore_index=True)
    
    # 중복 제거
    if 'ID' in df_combined.columns and 'Type' in df_combined.columns and 'Source' in df_combined.columns:
        df_combined = df_combined[df_combined['ID'].notna() & (df_combined['ID'] != '')]
        df_combined = df_combined.drop_duplicates(subset=['ID', 'Type', 'Source'])

    logger.info(f"Total collected unique data count: {len(df_combined)}")
    return df_combined

def run_analysis_pipeline(data_manager: DataManager,
                          nara_client: Optional[NaraJangteoApiClient], 
                          dapa_client: Optional[DapaApiClient], 
                          gemini_analyzer: Optional[GeminiAnalyzer],
                          start_date: date, end_date: date, min_score: int, 
                          update_mode: str = 'none'):
    """전체 분석 파이프라인 실행 (증분 업데이트 및 신규 분석 적용)"""
    
    if not data_manager:
        return classify_results(pd.DataFrame())

    # 1. 데이터 수집 (update_mode에 따라 범위 결정)
    if update_mode != 'none' and (nara_client or dapa_client):
        logger.info(f"--- Starting Data Collection Phase (Mode: {update_mode}) ---")
        
        # 수집 기간 설정
        if update_mode == 'incremental':
            # 성능 개선: 빠른 업데이트 (최근 7일 데이터만 수집)
            collection_end = date.today()
            collection_start = collection_end - timedelta(days=7)
            logger.info(f"Incremental mode: Collecting data from {collection_start} to {collection_end}")
        else: 
            # 전체 기간 재스캔: 사용자가 설정한 기간 전체 수집
            collection_start = start_date
            collection_end = end_date
            logger.info(f"Full scan mode: Collecting data from {collection_start} to {collection_end}")

        df_collected_raw = collect_data(nara_client, dapa_client, collection_start, collection_end)
        
        if not df_collected_raw.empty:
            # 2. 데이터 저장 및 신규 데이터 식별 (성능 개선: 핵심)
            df_collected, new_uids = data_manager.save_and_identify_new_data(df_collected_raw)
            
            # 3. 분석 실행 (신규 데이터에 대해서만 실행)
            if new_uids:
                logger.info("--- Starting Analysis Phase (Only on NEW data) ---")
                # 신규 데이터만 필터링
                df_new_data = df_collected[df_collected['UID'].isin(new_uids)].copy()
                
                keywords_set = load_keywords()
                
                # 3-1. 키워드 기반 분석 (신규 데이터 전체, 벡터화 엔진 사용)
                # min_score=0으로 설정하여 모든 점수 계산 및 저장
                df_analyzed = analyze_data_vectorized(df_new_data, keywords_set, min_score=0) 

                # 3-2. AI 심층 분석 (신규 데이터 중 관심 점수 이상)
                if not df_analyzed.empty:
                    # AI 분석은 사용자가 설정한 최소 점수 기준을 만족하는 항목에 대해서만 실행
                    df_high_score = df_analyzed[df_analyzed['Score'] >= min_score].copy()
                    
                    if gemini_analyzer and gemini_analyzer.model and not df_high_score.empty:
                        df_ai_results = gemini_analyzer.run_batch_analysis(df_high_score)
                        
                        # AI 결과 병합
                        if 'UID' in df_analyzed.columns and 'UID' in df_ai_results.columns:
                            df_ai_data = df_ai_results[['UID', 'AI_Analysis', 'AI_Strategy']]
                            if 'AI_Analysis' in df_analyzed.columns:
                                df_analyzed = df_analyzed.drop(columns=['AI_Analysis', 'AI_Strategy'], errors='ignore')
                            df_analyzed = pd.merge(df_analyzed, df_ai_data, on='UID', how='left')

                    # 4. 분석 결과 저장
                    data_manager.save_analysis_results(df_analyzed)
            else:
                logger.info("No new data to analyze.")
        else:
            logger.info("No data collected.")
    else:
        logger.info("Skipping collection phase. Loading existing data.")

    # 5. 최종 결과 로드 (DB에서 사용자가 요청한 조건에 맞는 데이터 로드)
    logger.info("--- Loading Final Results ---")
    df_final = data_manager.load_data(start_date, end_date, min_score)
    
    results = classify_results(df_final)
    logger.info("Analysis pipeline complete.")
    return results

def classify_results(df: pd.DataFrame):
    """데이터프레임을 조달 단계별로 분류합니다."""
    results = {
        'OrderPlans': pd.DataFrame(),
        'PriorStandards': pd.DataFrame(),
        'BidNotices': pd.DataFrame(),
        'Contracts': pd.DataFrame()
    }
    
    if not df.empty:
        df = df.sort_values(by='Score', ascending=False)
        results['OrderPlans'] = df[df['Type'].str.contains("발주계획|조달계획", na=False)].copy()
        results['PriorStandards'] = df[df['Type'].str.contains("사전규격|사전예고", na=False)].copy()
        results['BidNotices'] = df[df['Type'].str.contains("입찰공고", na=False)].copy()
        results['Contracts'] = df[df['Type'].str.contains("계약현황", na=False)].copy()
    return results