"""
AI 기반 국방/경찰 조달 정보 분석 대시보드
Version: 14.2 - 캐싱 전략 수정 및 안정성 확보 버전
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import logging
import os

# analyzer 모듈 import
try:
    from analyzer import (
        run_analysis_pipeline,
        NaraJangteoApiClient,
        DapaApiClient,
        GeminiAnalyzer,
        GEMINI_AVAILABLE,
        DataManager,
        classify_results # classify_results 임포트 추가
    )
except ImportError as e:
    st.error(f"analyzer.py 모듈을 임포트할 수 없습니다. 파일이 존재하는지 확인하세요. (Error: {e})")
    st.stop()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ============================================
# 페이지 설정 및 스타일
# ============================================

st.set_page_config(
    page_title="🎯 조달 정보 통합 분석 시스템 v14.2 (안정화됨)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #003366; margin-bottom: 20px; }
    div[data-testid="stMetric"] {
        background-color: #E6F0FF; border: 1px solid #B3D1FF; padding: 15px;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; padding: 10px 20px; border-radius: 5px 5px 0 0;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #003366; color: white; }
    .ai-card {
        border: 1px solid #ccc; border-radius: 8px; padding: 15px; margin-bottom: 15px;
        background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .ai-card-title { font-size: 1.2rem; font-weight: bold; color: #003366; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 세션 상태 및 데이터 관리자 초기화
# ============================================

DB_PATH = 'procurement_data.db'

# 성능 개선: @st.cache_resource 사용 (앱 실행 중 DataManager 인스턴스 유지)
@st.cache_resource
def get_data_manager(db_path):
    try:
        return DataManager(db_path=db_path)
    except RuntimeError as e:
        logging.error(f"Failed to initialize DataManager: {e}")
        return None

data_manager = get_data_manager(DB_PATH)

if data_manager is None:
    st.error("🚨 데이터베이스 초기화 실패. 서버 로그를 확인하고 권한 및 디스크 공간을 점검하세요.")
    st.stop()

# 세션 상태 초기화 (이 블록을 통째로 복사해서 붙여넣으세요)
if 'api_key_nara' not in st.session_state:
    st.session_state.api_key_nara = ""
if 'api_key_dapa' not in st.session_state:
    st.session_state.api_key_dapa = ""
if 'api_key_gemini' not in st.session_state:
    st.session_state.api_key_gemini = ""
if 'min_score' not in st.session_state:
    st.session_state.min_score = 20
if 'last_update' not in st.session_state:
    st.session_state.last_update = data_manager.get_last_collection_date()


# ============================================
# 유틸리티 함수 (캐싱 전략 수정됨)
# ============================================

# 성능 개선: @st.cache_data 적용 (DB에서 로드된 원본 통합 DataFrame 캐싱)
@st.cache_data(ttl=3600) # 1시간 동안 캐시 유지
def load_combined_data_from_db(start_date, end_date, min_score):
    """DB에서 통합된 DataFrame을 로드하고 결과를 캐싱합니다."""
    logging.info(f"Cache miss: Loading combined DataFrame from DB for {start_date}~{end_date}, Score>={min_score}")
    # data_manager.load_data()를 직접 호출하여 단일 DataFrame을 반환 및 캐싱
    return data_manager.load_data(start_date, end_date, min_score)

def execute_data_update(nara_key, dapa_key, gemini_key, start_date, end_date, min_score, update_mode):
    """데이터 수집 및 분석을 실행합니다 (update_mode='incremental' or 'full')."""
    
    # 클라이언트 및 분석기 초기화
    nara_client = NaraJangteoApiClient(nara_key) if nara_key else None
    dapa_client = DapaApiClient(dapa_key) if dapa_key else None
    gemini_analyzer = None
    if gemini_key and GEMINI_AVAILABLE:
        try:
            gemini_analyzer = GeminiAnalyzer(gemini_key)
        except Exception as e:
            logging.error(f"Gemini Analyzer 초기화 실패: {e}")
            st.warning("AI 분석기 초기화 실패. AI 분석은 제외됩니다.")

    # 파이프라인 실행 (DB 업데이트 수행)
    # run_analysis_pipeline은 내부적으로 DB 업데이트를 처리함 (결과 반환 안함)
    run_analysis_pipeline(
        data_manager, nara_client, dapa_client, gemini_analyzer, 
        start_date, end_date, min_score, update_mode=update_mode
    )
    
    # 중요: 업데이트 후 캐시 초기화 (새 데이터 반영을 위함)
    load_combined_data_from_db.clear() 
    st.session_state.last_update = data_manager.get_last_collection_date()
    # 결과는 반환하지 않음. 메인 흐름에서 최신 데이터를 다시 로드함.

# 날짜 포맷팅 유틸리티 (Streamlit 표시용)
def format_date_for_display(date_str):
    if not date_str or pd.isna(date_str): return "N/A"
    try:
        # DB에는 표준 ISO 형식(YYYY-MM-DD HH:MM:SS)으로 저장되어 있음
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        # 시간이 00:00:00이면 날짜만 표시
        if dt.time() == datetime.min.time():
            return dt.strftime('%Y-%m-%d')
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(date_str)

# 공통 데이터프레임 표시 설정
def display_dataframe(df, data_type):
    df_display = df.copy()
    # 날짜 컬럼 포맷팅 적용
    date_cols = ['AnnouncementDate', 'Deadline', 'OpeningDate', 'ContractDate']
    for col in date_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_date_for_display)
    
    if 'Budget' in df_display.columns:
        df_display['Budget'] = pd.to_numeric(df_display['Budget'], errors='coerce').fillna(0)

    base_cols = ['Score', 'Title', 'Source', 'Agency', 'Budget']
    if data_type == 'Plans':
        base_cols.extend(['AnnouncementDate'])
    elif data_type == 'Priors':
        base_cols.extend(['AnnouncementDate', 'Deadline'])
    elif data_type == 'Bids':
        base_cols.extend(['AnnouncementDate', 'Deadline', 'OpeningDate'])
    elif data_type == 'Contracts':
        base_cols.insert(4, 'Winner')
        base_cols.extend(['ContractDate'])

    base_cols.extend(['Type', 'Details', 'Link', 'MatchedKeywords'])
    cols_to_show = [col for col in base_cols if col in df_display.columns]
    
    st.dataframe(
        df_display[cols_to_show],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("링크/규격서", display_text="바로가기"),
            "Score": st.column_config.NumberColumn("점수", format="%d ⭐"),
            "Budget": st.column_config.NumberColumn("금액/예산 (원)", format="%,d"),
            "Source": st.column_config.TextColumn("출처"),
            "AnnouncementDate": st.column_config.TextColumn("공고(예정)일"),
            "Deadline": st.column_config.TextColumn("마감일시"),
            "OpeningDate": st.column_config.TextColumn("개찰일시"),
            "ContractDate": st.column_config.TextColumn("계약일"),
        }
    )

# ============================================
# 사이드바 (설정 및 실행)
# ============================================

with st.sidebar:
    st.header("🔑 API 설정")
    st.session_state.api_key_nara = st.text_input("나라장터 API 키", value=st.session_state.api_key_nara, type="password")
    st.session_state.api_key_dapa = st.text_input("방위사업청 API 키", value=st.session_state.api_key_dapa, type="password")
    st.session_state.api_key_gemini = st.text_input("Gemini API 키 (AI 분석용)", value=st.session_state.api_key_gemini, type="password")

    st.header("📅 조회 기간 설정")
    today = date.today()
    default_start = today - timedelta(days=30)

    col_start, col_end = st.columns(2)
    # 사용자가 설정한 날짜는 전역 변수로 사용됨
    start_date = col_start.date_input("시작일", default_start)
    end_date = col_end.date_input("종료일", today)

    st.header("⚙️ 분석 옵션")
    st.session_state.min_score = st.slider("최소 관심 점수", 0, 200, st.session_state.min_score)

    # 실행 버튼 로직 (수정됨)
    st.subheader("🔄 데이터 관리")
    if st.session_state.last_update:
        st.success(f"마지막 DB 업데이트:\n{st.session_state.last_update}")
    else:
        st.warning("저장된 데이터 없음")

    # 1. 데이터 로드/필터링 버튼
    # 버튼 클릭 시 Streamlit 스크립트가 재실행되며, 메인 로직에서 현재 설정값(날짜, 점수)으로 데이터를 로드함
    if st.button("🔎 설정 적용 및 데이터 로드 (빠름)", use_container_width=True):
        st.success("설정이 적용되었습니다. 데이터를 로드합니다.")

    st.markdown("---")
    st.subheader("🚀 최신 데이터 업데이트")

    # 2. 빠른 업데이트 (증분 수집)
    if st.button("⚡ 빠른 업데이트 (최근 7일)", use_container_width=True):
        if not st.session_state.api_key_nara and not st.session_state.api_key_dapa:
            st.error("API 키를 입력해야 합니다.")
        else:
            spinner_text = "최근 7일 데이터 수집 및 신규 항목 분석 중... (고성능 엔진 적용)"
            with st.spinner(spinner_text):
                try:
                    # 업데이트 실행 (캐시 초기화 포함)
                    execute_data_update(
                        st.session_state.api_key_nara, st.session_state.api_key_dapa, st.session_state.api_key_gemini,
                        start_date, end_date, st.session_state.min_score, update_mode='incremental'
                    )
                    st.success("빠른 업데이트 완료! 최신 데이터를 로드합니다.")
                except Exception as e:
                    st.error(f"업데이트 중 오류 발생: {e}")
                    logging.exception("Incremental update error")

    # 3. 전체 기간 재스캔 (전체 수집)
    if st.button("🔄 전체 기간 재스캔 (느림)", use_container_width=True):
        if not st.session_state.api_key_nara and not st.session_state.api_key_dapa:
            st.error("API 키를 입력해야 합니다.")
        else:
            spinner_text = "전체 기간 데이터 수집 및 신규 항목 분석 중... (시간 소요될 수 있음)"
            with st.spinner(spinner_text):
                try:
                    # 업데이트 실행 (캐시 초기화 포함)
                    execute_data_update(
                        st.session_state.api_key_nara, st.session_state.api_key_dapa, st.session_state.api_key_gemini,
                        start_date, end_date, st.session_state.min_score, update_mode='full'
                    )
                    st.success("전체 기간 재스캔 완료! 최신 데이터를 로드합니다.")
                except Exception as e:
                    st.error(f"재스캔 중 오류 발생: {e}")
                    logging.exception("Full scan error")

# ============================================
# 메인 대시보드 (데이터 흐름 수정됨)
# ============================================

st.markdown('<div class="main-title">🎯 조달 정보 통합 분석 시스템 v14.2 (안정화됨)</div>', unsafe_allow_html=True)

# 메인 데이터 로딩 흐름
combined_df = pd.DataFrame()
try:
    # 현재 설정값(사이드바의 날짜, 점수)을 기반으로 데이터 로드 시도 (캐시 활용)
    combined_df = load_combined_data_from_db(
        start_date, end_date, st.session_state.min_score
    )
except Exception as e:
    st.error(f"데이터 로드 중 심각한 오류 발생: {e}")
    logging.exception("Main data loading error")

# 데이터가 성공적으로 로드된 경우 처리
if not combined_df.empty:
    # 캐싱된 통합 데이터를 기반으로 실시간 분류 실행 (매우 빠름)
    results = classify_results(combined_df)

    # 분석 결과 요약 메트릭
    st.subheader("📊 전 주기 분석 요약 (나라장터 + 방위사업청)")
    
    counts = {
        'Plans': len(results.get('OrderPlans', pd.DataFrame())),
        'Priors': len(results.get('PriorStandards', pd.DataFrame())),
        'Bids': len(results.get('BidNotices', pd.DataFrame())),
        'Contracts': len(results.get('Contracts', pd.DataFrame()))
    }
    total_count = sum(counts.values())

    # AI 분석 여부 확인
    is_ai_analyzed = False
    # 통합 DataFrame에서 AI 분석 여부 확인
    if 'AI_Analysis' in combined_df.columns and combined_df['AI_Analysis'].notna().any():
            valid_ai = combined_df[combined_df['AI_Analysis'].notna() & (combined_df['AI_Analysis'] != '')]
            if not valid_ai.empty and not valid_ai['AI_Analysis'].str.contains("N/A|미실행|오류|미작동|실패|지원 불가|AI 엔진 미작동").all():
                is_ai_analyzed = True

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌟 총 결과", f"{total_count} 건")
    col2.metric("📋 1. 계획 단계", f"{counts['Plans']} 건")
    col3.metric("📑 2. 규격/예고 단계", f"{counts['Priors']} 건")
    col4.metric("📢 3. 공고 단계", f"{counts['Bids']} 건")
    col5.metric("🤝 4. 계약 단계", f"{counts['Contracts']} 건")

    st.markdown("---")

    # 탭 구조
    tabs = ["📋 1. 계획", "📑 2. 규격/예고", "📢 3. 공고", "🤝 4. 계약", "📈 통계"]
    if is_ai_analyzed:
        tabs.insert(0, "🤖 AI 심층 분석")

    tab_contents = st.tabs(tabs)
    tab_index = 0

    # 0. AI 심층 분석 탭
    if is_ai_analyzed:
        with tab_contents[tab_index]:
            st.subheader("🤖 AI 기반 사업 적합성 분석 및 전략 제안")
            
            # AI 분석 결과 필터링 (통합 DataFrame 사용)
            df_ai_results = combined_df[
                pd.notna(combined_df.get('AI_Analysis')) & 
                (combined_df['AI_Analysis'] != '') &
                ~combined_df['AI_Analysis'].str.contains("N/A|미실행|오류|미작동|실패|지원 불가|AI 엔진 미작동", na=False)
            ].copy()
            
            # 점수 순 정렬
            df_ai_results = df_ai_results.sort_values(by='Score', ascending=False)

            if not df_ai_results.empty:
                df_ai_results['Budget'] = pd.to_numeric(df_ai_results['Budget'], errors='coerce').fillna(0)
                
                for index, row in df_ai_results.iterrows():
                    st.markdown(f'<div class="ai-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ai-card-title">⭐{row["Score"]}점 | {row["Title"]}</div>', unsafe_allow_html=True)
                    
                    col_info1, col_info2, col_info3 = st.columns([1, 2, 2])
                    col_info1.metric("출처/기관", f"{row['Source']} / {row.get('Agency', 'N/A')}")
                    
                    budget_value = row.get('Budget')
                    budget_str = f"{float(budget_value):,.0f} 원" if budget_value and budget_value > 0 else "정보 없음"
                    col_info2.metric("예산/금액", budget_str)
                    
                    # 중요 날짜 표시 (마감일 우선, 표준화된 날짜 사용)
                    date_label = "주요 일정"; date_info = "N/A"
                    if row.get('Deadline') and not pd.isna(row.get('Deadline')):
                        date_label = "마감 일시"; date_info = format_date_for_display(row['Deadline'])
                    elif row.get('OpeningDate') and not pd.isna(row.get('OpeningDate')):
                        date_label = "개찰 일시"; date_info = format_date_for_display(row['OpeningDate'])
                    elif row.get('ContractDate') and not pd.isna(row.get('ContractDate')):
                         date_label = "계약일"; date_info = format_date_for_display(row['ContractDate'])
                    elif row.get('AnnouncementDate') and not pd.isna(row.get('AnnouncementDate')):
                        date_label = "공고(예정)일"; date_info = format_date_for_display(row['AnnouncementDate'])
                        
                    col_info3.metric(date_label, date_info)

                    st.markdown("---")

                    col_ai1, col_ai2 = st.columns(2)
                    with col_ai1:
                        st.markdown("**🔍 분석 요약 (Analysis Summary)**")
                        st.write(row['AI_Analysis'])
                    with col_ai2:
                        st.markdown("**💡 전략 제안 (Strategy Proposal)**")
                        st.write(row.get('AI_Strategy', 'N/A'))
                    
                    with st.expander("세부 정보 및 링크"):
                        st.text(f"유형: {row['Type']}")
                        if row.get('Link'):
                            st.markdown(f"[링크 바로가기]({row['Link']})", unsafe_allow_html=True)
                        st.text(f"상세 내용:\n{row.get('Details', '')}")

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("표시할 유효한 AI 분석 결과가 없습니다.")
        tab_index += 1

    # 1. 계획 단계 탭
    with tab_contents[tab_index]:
        st.subheader("📋 발주 및 조달 계획 상세")
        df_plan = results.get('OrderPlans')
        if df_plan is not None and not df_plan.empty:
            display_dataframe(df_plan, 'Plans')
        else:
            st.warning("관련 계획 정보가 없습니다.")
    tab_index += 1

    # 2. 규격/예고 단계 탭
    with tab_contents[tab_index]:
        st.subheader("📑 사전 규격 및 사전 예고 상세")
        df_prior = results.get('PriorStandards')
        if df_prior is not None and not df_prior.empty:
            display_dataframe(df_prior, 'Priors')
        else:
            st.warning("관련 사전규격/예고 정보가 없습니다.")
    tab_index += 1

    # 3. 공고 단계 탭
    with tab_contents[tab_index]:
        st.subheader("📢 입찰 공고 상세")
        df_bid = results.get('BidNotices')
        if df_bid is not None and not df_bid.empty:
            display_dataframe(df_bid, 'Bids')
        else:
            st.warning("관련 입찰공고 정보가 없습니다.")
    tab_index += 1

    # 4. 계약 단계 탭
    with tab_contents[tab_index]:
        st.subheader("🤝 계약 현황 상세")
        df_contract = results.get('Contracts')
        if df_contract is not None and not df_contract.empty:
             display_dataframe(df_contract, 'Contracts')
        else:
            st.warning("관련 계약 정보가 없습니다.")
    tab_index += 1

    # 5. 통계/시각화 탭
    with tab_contents[tab_index]:
        st.subheader("📈 통합 통계 및 시각화")
        
        # 통합 DataFrame(combined_df)을 사용하여 시각화
        if not combined_df.empty:
            col_stat1, col_stat2 = st.columns(2)

            with col_stat1:
                # 출처별 분포
                st.plotly_chart(px.pie(combined_df, names='Source', title='출처별 데이터 분포', hole=0.3, color_discrete_map={'나라장터':'#1f77b4', '방위사업청':'#ff7f0e'}), use_container_width=True)
            
            with col_stat2:
                # 조달 단계별 분포 시각화
                def categorize_stage(type_str):
                    if "계획" in type_str: return "1. 계획"
                    if "사전규격" in type_str or "사전예고" in type_str: return "2. 규격/예고"
                    if "입찰공고" in type_str: return "3. 공고"
                    if "계약현황" in type_str: return "4. 계약"
                    return "기타"
                
                # SettingWithCopyWarning 방지를 위해 .copy() 사용
                plot_df = combined_df.copy()
                plot_df['Stage'] = plot_df['Type'].apply(categorize_stage)
                
                fig = px.histogram(plot_df, x='Stage', title='조달 단계별 건수', color='Source', barmode='group', 
                                   category_orders={"Stage": ["1. 계획", "2. 규격/예고", "3. 공고", "4. 계약"]},
                                   color_discrete_map={'나라장터':'#1f77b4', '방위사업청':'#ff7f0e'})
                st.plotly_chart(fig, use_container_width=True)
            
            # 세부 유형별 분포
            fig_type = px.histogram(combined_df, x='Type', title='세부 유형별 건수', color='Source', color_discrete_map={'나라장터':'#1f77b4', '방위사업청':'#ff7f0e'})
            fig_type.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_type, use_container_width=True)

        else:
            st.info("시각화할 데이터가 없습니다.")

else:
    # 데이터가 비어있는 경우 (초기 상태 또는 로드 실패 시)
    st.info("⬅️ 사이드바에서 설정을 적용하거나 데이터를 업데이트해 주세요.")
    st.success("✅ 시스템 최적화 및 캐시 안정화 완료.")