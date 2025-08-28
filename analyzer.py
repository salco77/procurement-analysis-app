"""
AI 기반 국방/경찰 조달 정보 분석 시스템
Version: 12.0 - AI 기반 완전 통합 버전
"""

import os
import requests
import requests.adapters
import urllib3
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime, timedelta, date
# 월 단위 분할을 위해 dateutil 사용 (설치 필요: pip install python-dateutil)
from dateutil.relativedelta import relativedelta 
import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import xmltodict
from typing import Optional, Dict, List, Any

# Gemini AI SDK (설치 필요: pip install google-generativeai)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai library not found. AI analysis will be disabled.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# 1. 회사 프로필 및 키워드 설정
# ============================================
# 참고: 실제 운영 시에는 이전 버전(v8.1)의 상세 키워드 정의를 그대로 사용하세요.

COMPANY_PROFILE = """
우리 회사는 국방/경찰 분야의 종합 솔루션 제공 기업으로 다음 사업을 수행합니다:
[1. VR/AR/XR 훈련 시뮬레이션 사업]
- 영상 모의 사격 훈련 시스템 및 과학화 사격장, 비행/항공/헬기 시뮬레이터, 전차/장갑차 운용 시뮬레이터 개발 및 납품 경험 다수 보유.
[2. 무기체계 정비/유지보수 사업 (MRO)]
- 유도무기 체계(비궁, 천무 등) 및 화력장비(K9, K2) 정비 지원 시스템 개발 및 MRO 수행.
[3. 첨단기술 국방 솔루션]
- AI 기반 훈련평가 및 예측정비, IoT/빅데이터 기반 장비 관리 시스템 연구개발.
"""

# 키워드 정의 예시
SIMULATION_CORE_KEYWORDS = {"시뮬레이터", "모의훈련", "VR", "XR", "과학화사격장", "영상사격", "가상현실"}
WEAPON_CORE_KEYWORDS = {"비궁", "천무", "유도무기", "K9", "K2", "현무"}
MRO_KEYWORDS = {"정비", "유지보수", "MRO", "PBL", "성능개량", "창정비"}
ADVANCED_TECH_KEYWORDS = {"AI", "빅데이터", "드론", "IoT", "예측정비", "인공지능"}
GENERAL_KEYWORDS = {"시스템", "체계", "개발", "연구용역", "솔루션"}
BROAD_KEYWORDS = {"국방", "경찰", "연구", "훈련", "사업"}
NEGATIVE_KEYWORDS = {"가상계좌", "모의고사", "토목", "건축", "급식", "사무용품", "인쇄"}

# 키워드 가중치 설정
KEYWORD_WEIGHTS = {
    "핵심_시뮬레이션": 15, "핵심_무기체계": 15, "MRO": 10, "첨단기술": 8, "일반": 5, "광범위": 1
}

# ============================================
# 2. 키워드 관리 및 분석 함수
# ============================================

def load_keywords():
    """키워드를 로드합니다."""
    return {
        "핵심_시뮬레이션": SIMULATION_CORE_KEYWORDS,
        "핵심_무기체계": WEAPON_CORE_KEYWORDS,
        "MRO": MRO_KEYWORDS,
        "첨단기술": ADVANCED_TECH_KEYWORDS,
        "일반": GENERAL_KEYWORDS,
        "광범위": BROAD_KEYWORDS,
        "제외": NEGATIVE_KEYWORDS
    }

def save_keywords(keywords):
    logger.info("Keywords saved (placeholder).")

def calculate_score(text, keywords):
    """텍스트와 키워드를 기반으로 점수를 계산합니다."""
    if not text:
        return 0, []

    text_lower = str(text).lower()
    score = 0
    matched = set()

    # 제외 키워드 확인
    for neg_kw in keywords.get("제외", []):
        if neg_kw.lower() in text_lower:
            return -100, [f"제외: {neg_kw}"]

    # 키워드 매칭 및 가중치 적용
    for category, kws in keywords.items():
        if category == "제외":
            continue
        weight = KEYWORD_WEIGHTS.get(category, 1)
        for kw in kws:
            if kw.lower() in text_lower:
                score += weight
                if category != "광범위":
                    matched.add(kw)

    return score, sorted(list(matched))

def analyze_data(df, keywords, min_score):
    """데이터프레임 전체를 분석하고 점수를 계산합니다."""
    if df.empty:
        return pd.DataFrame()

    analyzed_data = []
    for index, row in df.iterrows():
        # 제목과 상세 내용을 합쳐서 분석
        text_to_analyze = f"{row.get('Title', '')}\n{row.get('Details', '')}"
        score, matched_keywords = calculate_score(text_to_analyze, keywords)

        if score >= min_score:
            row_data = row.to_dict()
            row_data['Score'] = score
            row_data['MatchedKeywords'] = ", ".join(matched_keywords)
            analyzed_data.append(row_data)

    if not analyzed_data:
        return pd.DataFrame()

    df_analyzed = pd.DataFrame(analyzed_data)
    # 중복 제거 재확인
    if 'ID' in df_analyzed.columns and 'Type' in df_analyzed.columns:
         df_analyzed = df_analyzed.drop_duplicates(subset=['ID', 'Type'])

    return df_analyzed.sort_values(by='Score', ascending=False)

# ============================================
# 3. AI 분석 엔진 (Gemini 연동)
# ============================================

class GeminiAnalyzer:
    """Gemini AI를 사용하여 조달 정보를 심층 분석합니다."""
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            self.model = None
            return
        try:
            genai.configure(api_key=api_key)
            # 속도와 비용 효율을 위해 gemini-1.5-flash 사용
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini Analyzer initialized successfully.")
        except Exception as e:
            logger.error(f"Gemini AI 초기화 실패: {e}")
            self.model = None

    def generate_prompt(self, item: Dict[str, Any]):
        """분석을 위한 프롬프트를 생성합니다."""
        # 상세 내용은 1500자로 제한하여 토큰 최적화
        details_short = str(item.get('Details', ''))[:1500]

        prompt = f"""
        당신은 국방 조달 전문 분석가입니다. 아래 제공된 '회사 프로필'을 기반으로 '조달 정보'가 해당 회사에 얼마나 적합한지 분석하고 전략을 제안해야 합니다.

        [회사 프로필]
        {COMPANY_PROFILE}

        [조달 정보]
        유형: {item.get('Type')}
        제목: {item.get('Title')}
        기관: {item.get('Agency')}
        예산/금액: {item.get('Budget', 0):,.0f} 원
        상세 내용 및 품목:
        {details_short}...

        [분석 지침]
        1. 회사의 핵심 역량(시뮬레이션, MRO, 첨단기술)과 조달 정보의 연관성을 명확히 분석합니다.
        2. 사업 참여 시 강점과 잠재적 기회를 구체적으로 언급합니다.
        3. 구체적인 사업 참여 전략(예: 단독 참여 가능성, 필수 협력 분야, 기술 제안 방향)을 제안합니다.
        4. 응답은 반드시 아래 JSON 형식으로 제공해야 합니다. 다른 설명은 생략하고 JSON만 출력하세요.

        [응답 형식 (JSON)]
        {{
            "analysis_summary": "회사의 A 역량을 활용하여 B 요구사항을 충족할 수 있음. 특히 C 경험이 강점으로 작용할 것임.",
            "strategy_proposal": "단독 참여를 목표로 하되, D 기술 분야는 협력 고려. F 방향으로 기술 제안서 강조 필요."
        }}
        """
        return prompt

    def analyze_item(self, item: Dict[str, Any]):
        """단일 항목을 분석합니다."""
        if not self.model:
            return {"AI_Analysis": "AI 엔진 미작동", "AI_Strategy": "N/A"}

        prompt = self.generate_prompt(item)
        try:
            # AI 호출 설정 (안정성 위주)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.3)
            )
            # 응답에서 JSON 추출 및 파싱
            response_text = response.text.strip()
            # Markdown 코드 블록 제거 (```json ... ```)
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                 response_text = response_text[3:-3].strip()
            
            result = json.loads(response_text)
            return {
                "AI_Analysis": result.get("analysis_summary", "분석 실패"),
                "AI_Strategy": result.get("strategy_proposal", "제안 실패")
            }
        except json.JSONDecodeError:
            logger.error(f"AI 응답 JSON 파싱 실패: {item.get('ID')} - 응답 샘플: {response.text[:100]}")
            return {"AI_Analysis": f"AI 응답 형식 오류: {response.text[:50]}...", "AI_Strategy": "N/A"}
        except Exception as e:
            logger.error(f"AI 분석 중 오류 발생: {item.get('ID')} - {e}")
            return {"AI_Analysis": f"분석 오류: {e}", "AI_Strategy": "N/A"}

    def run_batch_analysis(self, df: pd.DataFrame):
        """데이터프레임 전체를 병렬로 분석합니다."""
        if not self.model or df.empty:
            # AI 분석기가 없으면 빈 컬럼 추가하여 반환
            df['AI_Analysis'] = "N/A (AI 미실행 또는 지원 불가)"
            df['AI_Strategy'] = "N/A"
            return df

        logger.info(f"Starting AI deep dive analysis on {len(df)} items...")
        results = []
        
        # AI 호출은 시간이 걸리므로 병렬 처리 (동시 요청 수 10개 제한)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # DataFrame의 인덱스를 사용하여 결과 매핑
            future_to_index = {executor.submit(self.analyze_item, row.to_dict()): index for index, row in df.iterrows()}
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append({'index': index, **result})
                except Exception as e:
                    logger.error(f"AI batch processing error at index {index}: {e}")
                    results.append({'index': index, "AI_Analysis": "처리 중 오류", "AI_Strategy": "N/A"})

        # 결과를 원본 데이터프레임에 병합
        df_ai = pd.DataFrame(results).set_index('index')
        # 원본 DataFrame의 인덱스를 기준으로 병합 (원본 인덱스 유지)
        df_merged = df.join(df_ai, how='left')
        logger.info("AI analysis complete.")
        return df_merged


# ============================================
# 4. API 클라이언트 정의
# ============================================

class BaseApiClient:
    """API 호출을 위한 기본 클래스"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _call_api(self, url: str, params: Dict[str, Any], response_format: str = 'json'):
        """API 호출 및 응답 처리"""
        if not self.api_key: return None
        params['ServiceKey'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30, verify=False)
            response.raise_for_status()

            if response_format == 'json':
                return response.json()
            elif response_format == 'xml':
                response.encoding = 'utf-8'
                return xmltodict.parse(response.text)
        except Exception as e:
            logger.error(f"API 호출 실패: {url} - Error: {e}")
            return None

class NaraJangteoApiClient(BaseApiClient):
    """조달청 나라장터 API 클라이언트 (JSON 기반)"""
    BASE_URL = "http://apis.data.go.kr/1230000/ao"
    PAGE_SIZE = 500

    def _generate_monthly_ranges(self, start_date, end_date):
        """API의 1개월 조회 제한 대응을 위해 달력 기준 월 단위로 분할합니다."""
        ranges = []
        current_start = start_date
        while current_start <= end_date:
            # 해당 월의 마지막 날 계산
            month_end = current_start + relativedelta(day=31)
            # 실제 종료일과 비교하여 더 이른 날짜를 사용
            current_end = min(month_end, end_date)
            ranges.append((current_start, current_end))
            # 다음 달 1일로 시작일 설정
            current_start = current_end + timedelta(days=1)
        return ranges

    def _fetch_paginated_data(self, endpoint, params_base, type_name=None):
        """JSON 페이지네이션 처리 공통 로직 (Robustness 강화)"""
        all_data = []
        page_no = 1
        while True:
            params = params_base.copy()
            params.update({'pageNo': page_no, 'numOfRows': self.PAGE_SIZE, 'type': 'json'})
            data = self._call_api(self.BASE_URL + endpoint, params, 'json')

            if data and data.get('response') and data['response']['header'].get('resultCode') == '00':
                body = data['response']['body']
                
                # API 응답 구조 변형 대응 로직
                items = body.get('items')
                if isinstance(items, dict) and 'item' in items:
                     items = items.get('item')
                if items is None:
                    items = body.get('item')

                if not items: break

                if not isinstance(items, list):
                    items = [items]

                for item in items:
                    # type_name이 제공되면 저장, 없으면 API 응답의 업무구분(bsnsDivNm) 사용 시도
                    item['_type_name'] = type_name or item.get('bsnsDivNm')
                    all_data.append(item)

                total_count = int(body.get('totalCount', 0))
                if page_no * self.PAGE_SIZE >= total_count: break
                page_no += 1
            else:
                logger.warning(f"NaraJangteo API ({endpoint}) 응답 오류 또는 데이터 없음")
                break
        return all_data

    # --- 1. 발주계획 (OrderPlanSttusService) ---
    def get_order_plans(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Order Plans...")
        start_ym = start_date.strftime('%Y%m'); end_ym = end_date.strftime('%Y%m')
        endpoints = {"물품": "/OrderPlanSttusService/getOrderPlanSttusListThng", "공사": "/OrderPlanSttusService/getOrderPlanSttusListCnstwk", "용역": "/OrderPlanSttusService/getOrderPlanSttusListServc"}
        all_items = []
        params_base = {'inqryDiv': '1', 'orderBgnYm': start_ym, 'orderEndYm': end_ym}

        for type_name, endpoint in endpoints.items():
            items = self._fetch_paginated_data(endpoint, params_base, type_name)
            all_items.extend(items)

        normalized_data = []
        for item in all_items:
            details = (f"발주월: {item.get('orderMnth')}, 계약방법: {item.get('cntrctMthdNm')}\n"
                       f"담당자: {item.get('ofclNm')}\n"
                       f"세부품명/규격: {item.get('dtilPrdctClsfcNoNm')} / {item.get('specCntnts')}")
            normalized_item = {
                'ID': item.get('orderPlanUntyNo'), 'Title': item.get('bizNm'),
                'Date': item.get('nticeDt') or f"{item.get('orderYear')}-{item.get('orderMnth')}",
                'Budget': float(item.get('sumOrderAmt') or 0), 'Type': f"발주계획 ({item.get('_type_name')})",
                'Source': '나라장터', 'Agency': item.get('orderInsttNm'),
                'Details': details, 'Link': None, 'Winner': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 2. 사전규격 (HrcspSsstndrdInfoService) ---
    def get_prior_standards(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Prior Standards...")
        start_dt_str = start_date.strftime('%Y%m%d0000'); end_dt_str = end_date.strftime('%Y%m%d2359')
        endpoints = {"물품": "/HrcspSsstndrdInfoService/getPublicPrcureThngInfoThng", "용역": "/HrcspSsstndrdInfoService/getPublicPrcureThngInfoServc", "공사": "/HrcspSsstndrdInfoService/getPublicPrcureThngInfoCnstwk"}
        all_items = []
        params_base = {'inqryDiv': '1', 'inqryBgnDt': start_dt_str, 'inqryEndDt': end_dt_str}

        for type_name, endpoint in endpoints.items():
            items = self._fetch_paginated_data(endpoint, params_base, type_name)
            all_items.extend(items)

        normalized_data = []
        for item in all_items:
            details = (f"담당자: {item.get('ofclNm')}, 마감일: {item.get('opninRgstClseDt')}\n"
                       f"품목상세: {item.get('prdctDtlList')}")
            normalized_item = {
                'ID': item.get('bfSpecRgstNo'), 'Title': item.get('prdctClsfcNoNm'), 'Date': item.get('rgstDt'),
                'Budget': float(item.get('asignBdgtAmt') or 0), 'Type': f"사전규격 ({item.get('_type_name')})",
                'Source': '나라장터', 'Agency': item.get('rlDminsttNm') or item.get('orderInsttNm'),
                'Details': details, 'Link': item.get('specDocFileUrl1'), 'Winner': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 3. 입찰공고 (PubDataOpnStdService) ---
    def get_bid_announcements(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Bid Announcements (Monthly chunks)...")
        endpoint = "/PubDataOpnStdService/getDataSetOpnStdBidPblancInfo"
        all_items = []
        date_ranges = self._generate_monthly_ranges(start_date, end_date)

        for chunk_start, chunk_end in date_ranges:
            params_base = {'bidNtceBgnDt': chunk_start.strftime('%Y%m%d0000'), 'bidNtceEndDt': chunk_end.strftime('%Y%m%d2359')}
            items = self._fetch_paginated_data(endpoint, params_base, None)
            all_items.extend(items)

        normalized_data = []
        for item in all_items:
            # 예산 결정: 배정예산(asignBdgtAmt) 또는 추정가격(estmtPrice) 사용 (필드명 추정)
            budget = float(item.get('asignBdgtAmt') or item.get('estmtPrice') or 0)
            details = (f"업무구분: {item.get('bsnsDivNm')}, 계약방법: {item.get('cntrctMthdNm')}, 공고상태: {item.get('bidNtceSttusNm')}\n"
                       f"개찰일시: {item.get('opengDate')} {item.get('opengTime')}\n담당자: {item.get('bidNtceChrgNm')}")
            agency = item.get('dminsttNm') or item.get('orderInsttNm') or item.get('ntceInsttNm')

            normalized_item = {
                'ID': f"{item.get('bidNtceNo')}-{item.get('bidNtceOrd')}", 'Title': item.get('bidNtceNm'),
                'Date': f"{item.get('bidNtceDate')} {item.get('bidNtceTime')}", 'Budget': budget,
                'Type': f"입찰공고 ({item.get('_type_name', '기타')})", 'Source': '나라장터', 'Agency': agency,
                'Details': details, 'Link': item.get('bidNtceUrl'), 'Winner': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 4. 계약현황 (PubDataOpnStdService) ---
    def get_contracts(self, start_date, end_date):
        logger.info("Fetching NaraJangteo Contracts (Monthly chunks)...")
        endpoint = "/PubDataOpnStdService/getDataSetOpnStdCntrctInfo"
        all_items = []
        date_ranges = self._generate_monthly_ranges(start_date, end_date)

        for chunk_start, chunk_end in date_ranges:
            # 계약일자(YYYYMMDD) 기준 조회 (필드명 추정)
            params_base = {'cntrctBgnDate': chunk_start.strftime('%Y%m%d'), 'cntrctEndDate': chunk_end.strftime('%Y%m%d')}
            items = self._fetch_paginated_data(endpoint, params_base, None)
            all_items.extend(items)

        normalized_data = []
        for item in all_items:
            # 총계약금액(totCntrctAmt) 또는 계약금액(cntrctAmt) (필드명 추정)
            budget = float(item.get('totCntrctAmt') or item.get('cntrctAmt') or 0)
            details = (f"업무구분: {item.get('bsnsDivNm')}, 계약방법: {item.get('cntrctMthdNm')}, 계약구분: {item.get('cntrctDivNm')}\n"
                       f"계약기간: {item.get('cntrctBeginDate')} ~ {item.get('cntrctEndDate')}")
            agency = item.get('dminsttNm') or item.get('orderInsttNm')
            winner = item.get('cntrctEntrpsNm') or item.get('sppirdNm')

            normalized_item = {
                'ID': item.get('cntrctNo'), 'Title': item.get('cntrctNm'), 'Date': item.get('cntrctDate'),
                'Budget': budget, 'Type': f"계약현황 ({item.get('_type_name', '기타')})",
                'Source': '나라장터', 'Agency': agency, 'Details': details,
                'Link': item.get('cntrctUrl'), 'Winner': winner
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)


class DapaApiClient(BaseApiClient):
    """방위사업청 API 클라이언트 (XML 기반)"""
    BASE_URL = "http://openapi.d2b.go.kr/openapi/service"
    PAGE_SIZE = 100

    def _fetch_paginated_data_xml(self, service, operation, params_base):
        """XML 페이지네이션 처리 공통 로직"""
        endpoint = f"{self.BASE_URL}{service}/{operation}"
        all_items = []
        page_no = 1
        while True:
            params = params_base.copy()
            params.update({'pageNo': page_no, 'numOfRows': self.PAGE_SIZE})
            data = self._call_api(endpoint, params, 'xml')

            if data and data.get('response') and data['response']['header'].get('resultCode') == '00':
                body = data['response']['body']
                items_container = body.get('items', {})
                items = items_container.get('item')
                
                if not items: break

                # XML 응답은 단일 항목일 때 리스트가 아님
                if not isinstance(items, list): items = [items]
                all_items.extend(items)

                total_count = int(body.get('totalCount', 0))
                if page_no * self.PAGE_SIZE >= total_count: break
                page_no += 1
            else:
                logger.warning(f"DAPA API ({operation}) 응답 오류 또는 데이터 없음")
                break
        return all_items

    # --- 1. 조달계획 및 사전규격 (PrcurePlanInfoService) ---
    def get_order_plans(self, start_date, end_date):
        logger.info("Fetching DAPA Order Plans (including Pre-Solicitation)...")
        start_ym = start_date.strftime('%Y%m'); end_ym = end_date.strftime('%Y%m')
        operations = {"국내": "getDmstcPrcurePlanList", "시설": "getFcltyPrcurePlanList"}
        all_items = []
        params_base = {'orderPrearngeMtBegin': start_ym, 'orderPrearngeMtEnd': end_ym}

        for type_name, operation in operations.items():
            items = self._fetch_paginated_data_xml("/PrcurePlanInfoService", operation, params_base)
            for item in items:
                item['_type_name'] = type_name
                # 필드명 통일화
                item['_title'] = item.get('cntrwkNm') or item.get('reprsntPrdlstNm')
                item['_id'] = item.get('cntrwkNo') or item.get('dcsNo')
            all_items.extend(items)

        normalized_data = []
        for item in all_items:
            # 사전규격공개 여부(beffatStndrdOthbcAt) 활용하여 단계 구분
            is_prior_std = item.get('beffatStndrdOthbcAt') == '공개'
            details = (f"진행상태: {item.get('progrsSttus')}, 계약방법: {item.get('cntrctMth')}, "
                       f"사전규격공개: {item.get('beffatStndrdOthbcAt', 'N/A')}")
            data_type = f"사전규격 ({item.get('_type_name')})" if is_prior_std else f"조달계획 ({item.get('_type_name')})"

            normalized_item = {
                'ID': f"DAPA-PLAN-{item.get('_id')}", 'Title': item.get('_title'), 'Date': item.get('orderPrearngeMt'),
                'Budget': float(item.get('budgetAmount') or 0), 'Type': data_type, 'Source': '방위사업청',
                'Agency': item.get('ornt'), 'Details': details, 'Link': "https://www.d2b.go.kr", 'Winner': None
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)

    # --- 2. 입찰공고 (BidPblancInfoService) ---
    def get_bid_announcements(self, start_date, end_date):
        logger.info("Fetching DAPA Bid Announcements (Deep Dive)...")
        start_dt_str = start_date.strftime('%Y%m%d'); end_dt_str = end_date.strftime('%Y%m%d')
        params_base = {'anmtDateBegin': start_dt_str, 'anmtDateEnd': end_dt_str}
        all_items = self._fetch_paginated_data_xml("/BidPblancInfoService", "getDmstcCmpetBidPblancList", params_base)
        
        # 상세 정보 및 품목 명세서로 보강 (병렬 처리)
        enriched_data = self._enrich_bid_data(all_items)
        return pd.DataFrame(enriched_data)

    def _enrich_bid_data(self, items):
        """입찰 공고 목록에 상세 정보와 품목 명세서를 병렬로 추가"""
        enriched_results = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_item = {executor.submit(self._fetch_bid_details_and_items, item): item for item in items}
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    if result: enriched_results.append(result)
                except Exception as e:
                    logger.error(f"공고 상세 정보 조회 중 오류: {e}")
        return enriched_results

    def _fetch_bid_details_and_items(self, item):
        SERVICE = "/BidPblancInfoService"
        params = {'demandYear': item.get('demandYear'), 'orntCode': item.get('orntCode'), 'dcsNo': item.get('dcsNo'), 'pblancNo': item.get('pblancNo'), 'pblancOdr': item.get('pblancOdr')}

        if not all([params['pblancNo'], params['pblancOdr']]): return None

        # 1. 상세 정보 조회
        detail_info = {'Budget': 0, 'EstimatedPrice': 0, 'Contact': 'N/A', 'DecisionMethod': 'N/A'}
        detail_data = self._call_api(f"{self.BASE_URL}{SERVICE}/getDmstcCmpetBidPblancDetail", params.copy(), 'xml')
        if detail_data and detail_data.get('response', {}).get('body', {}).get('item'):
             d_item = detail_data['response']['body']['item']
             detail_info['Budget'] = float(d_item.get('budgetAmount') or 0)
             detail_info['EstimatedPrice'] = float(d_item.get('estmPrce') or 0)
             detail_info['DecisionMethod'] = d_item.get('sucbidrDecsnMth')
             detail_info['Contact'] = f"{d_item.get('chargerNm')}"

        # 2. 품목 명세서 조회
        item_list_str = "N/A"
        if params['dcsNo'] and params['dcsNo'] != 'None':
            item_data = self._call_api(f"{self.BASE_URL}{SERVICE}/getDmstcCmpetBidPblancItem", params.copy(), 'xml')
            if item_data and item_data.get('response', {}).get('body', {}).get('items'):
                items_raw = item_data['response']['body']['items'].get('item')
                if items_raw:
                    if not isinstance(items_raw, list): items_raw = [items_raw]
                    item_details = [f"[{i.get('prdlstNm')}, 수량:{i.get('qy')}]" for i in items_raw]
                    item_list_str = "; ".join(item_details)

        # 예산 결정 및 정규화
        final_budget = detail_info['Budget'] or detail_info['EstimatedPrice'] or float(item.get('bsicExpt') or 0)
        details = (f"계약방법: {item.get('cntrctMth')}, 낙찰결정: {detail_info['DecisionMethod']}, 개찰일시: {item.get('opengDt')}\n"
                   f"담당자: {detail_info['Contact']}\n품목명세: {item_list_str}")

        return {
            'ID': f"{item.get('pblancNo')}-{item.get('pblancOdr')}", 'Title': item.get('bidNm'), 'Date': item.get('pblancDate'),
            'Budget': final_budget, 'Type': f"입찰공고 ({item.get('busiDivs', '국내')})", 'Source': '방위사업청',
            'Agency': item.get('ornt'), 'Details': details, 'Link': "https://www.d2b.go.kr", 'Winner': None
        }

    # --- 3. 계약현황 (CntrctInfoService) ---
    def get_contracts(self, start_date, end_date):
        logger.info("Fetching DAPA Contracts...")
        start_dt_str = start_date.strftime('%Y%m%d'); end_dt_str = end_date.strftime('%Y%m%d')
        operations = {"국내": "getDmstcCntrctInfoList", "시설": "getFcltyCntrctInfoList", "국외": "getOutnatnCntrctInfoList"}
        all_items = []
        params_base = {'cntrctDateBegin': start_dt_str, 'cntrctDateEnd': end_dt_str}

        for type_name, operation in operations.items():
            items = self._fetch_paginated_data_xml("/CntrctInfoService", operation, params_base)
            for item in items:
                item['_type_name'] = type_name
            all_items.extend(items)

        normalized_data = []
        for item in all_items:
            # 필드명 통일화
            title = item.get('cntrwkNm') or item.get('cntrctNm') or item.get('prdlstNm')
            contract_amount = float(item.get('cntrctAmnt') or item.get('cntrctAmount') or item.get('totCntrctAmount') or 0)
            winner = item.get('cntrctEntrpsNm') or item.get('cntrctCmpny') or item.get('entrpsNm')
            scsbid_rate = item.get('scsbidRt') or item.get('scsbidRate', 'N/A')

            details = (f"계약방법: {item.get('cntrctMth')}, 계약종류: {item.get('cntrctKnd', 'N/A')}, 낙찰률: {scsbid_rate}%")
            
            normalized_item = {
                'ID': item.get('cntrctNo'), 'Title': title, 'Date': item.get('cntrctDate'), 'Budget': contract_amount,
                'Type': f"계약현황 ({item.get('cntrctDivs', item.get('_type_name'))})", 'Source': '방위사업청',
                'Agency': item.get('ornt') or item.get('orderOrnt'), 'Details': details,
                'Link': "https://www.d2b.go.kr", 'Winner': winner
            }
            normalized_data.append(normalized_item)
        return pd.DataFrame(normalized_data)


# ============================================
# 5. 메인 분석 실행 함수 (통합 및 AI 연동)
# ============================================

def run_analysis(nara_client: Optional[NaraJangteoApiClient], 
                 dapa_client: Optional[DapaApiClient], 
                 gemini_analyzer: Optional[GeminiAnalyzer], # AI 분석기 추가
                 start_date: date, end_date: date, min_score=60):
    """전체 분석 프로세스를 실행합니다."""
    logger.info(f"Starting comprehensive analysis from {start_date} to {end_date}")

    keywords_set = load_keywords()
    all_data_frames = []

    # 1. 나라장터 데이터 수집 (전 주기)
    if nara_client:
        df_nara_plan = nara_client.get_order_plans(start_date, end_date)
        df_nara_prior = nara_client.get_prior_standards(start_date, end_date)
        df_nara_bid = nara_client.get_bid_announcements(start_date, end_date)
        df_nara_contract = nara_client.get_contracts(start_date, end_date)
        all_data_frames.extend([df_nara_plan, df_nara_prior, df_nara_bid, df_nara_contract])

    # 2. 방위사업청 데이터 수집 (전 주기)
    if dapa_client:
        df_dapa_plan = dapa_client.get_order_plans(start_date, end_date) # 사전규격 포함
        df_dapa_bid = dapa_client.get_bid_announcements(start_date, end_date)
        df_dapa_contract = dapa_client.get_contracts(start_date, end_date)
        all_data_frames.extend([df_dapa_plan, df_dapa_bid, df_dapa_contract])

    # 데이터 통합
    all_data_frames = [df for df in all_data_frames if df is not None and not df.empty]
    if not all_data_frames:
        logger.info("No data collected."); return {}

    # 통합 시 인덱스를 유지하여 AI 분석 결과 매핑에 사용
    df_combined = pd.concat(all_data_frames, ignore_index=True)
    
    # 중복 제거
    if 'ID' in df_combined.columns and 'Type' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['ID', 'Type'])

    # 3. 키워드 기반 분석
    df_analyzed = analyze_data(df_combined, keywords_set, min_score)

    # 4. AI 심층 분석
    # AI 분석은 키워드 필터링된 데이터에 대해서만 실행
    if gemini_analyzer and gemini_analyzer.model and not df_analyzed.empty:
        df_analyzed = gemini_analyzer.run_batch_analysis(df_analyzed)
    elif not df_analyzed.empty:
        # AI 분석기가 없거나 실패했을 경우 빈 컬럼 추가
        df_analyzed['AI_Analysis'] = "N/A (AI 미실행 또는 지원 불가)"
        df_analyzed['AI_Strategy'] = "N/A"


    # 결과 분류 (조달 단계별 4개 카테고리)
    results = {
        'OrderPlans': pd.DataFrame(),
        'PriorStandards': pd.DataFrame(),
        'BidNotices': pd.DataFrame(),
        'Contracts': pd.DataFrame()
    }
    
    if not df_analyzed.empty:
        results['OrderPlans'] = df_analyzed[df_analyzed['Type'].str.contains("발주계획|조달계획", na=False)].copy()
        results['PriorStandards'] = df_analyzed[df_analyzed['Type'].str.contains("사전규격", na=False)].copy()
        results['BidNotices'] = df_analyzed[df_analyzed['Type'].str.contains("입찰공고", na=False)].copy()
        results['Contracts'] = df_analyzed[df_analyzed['Type'].str.contains("계약현황", na=False)].copy()
    
    logger.info("Analysis complete.")
    return results