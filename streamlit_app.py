"""
AI 기반 국방/경찰 조달 정보 분석 대시보드
Version: 12.0 - AI 기반 완전 통합 대시보드
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import logging

# analyzer 모듈 import
# analyzer.py 파일이 동일 디렉토리에 있어야 합니다.
from analyzer import (
    run_analysis,
    load_keywords,
    NaraJangteoApiClient,
    DapaApiClient,
    GeminiAnalyzer, # AI 분석기 추가
    GEMINI_AVAILABLE
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ============================================
# 페이지 설정 및 스타일
# ============================================

st.set_page_config(
    page_title="🎯 조달 정보 통합 분석 시스템 v12.0 (AI)",
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
    /* AI 분석 카드 스타일 */
    .ai-card {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .ai-card-title { font-size: 1.1rem; font-weight: bold; color: #003366; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 세션 상태 초기화
# ============================================

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'api_key_nara' not in st.session_state:
    st.session_state.api_key_nara = ""
if 'api_key_dapa' not in st.session_state:
    st.session_state.api_key_dapa = ""
if 'api_key_gemini' not in st.session_state: # Gemini 키 추가
    st.session_state.api_key_gemini = ""
if 'min_score' not in st.session_state:
    st.session_state.min_score = 20 

# ============================================
# 유틸리티 함수 및 캐싱
# ============================================

# Streamlit 캐싱은 객체 인스턴스를 직접 캐싱하기 어려울 수 있으므로, 데이터만 캐싱하도록 합니다.
@st.cache_data(ttl=1800) # 30분 캐시
def run_cached_analysis(nara_key, dapa_key, gemini_key, start_date, end_date, min_score):
    """분석 실행 및 캐싱"""
    nara_client = NaraJangteoApiClient(nara_key) if nara_key else None
    dapa_client = DapaApiClient(dapa_key) if dapa_key else None
    
    # Gemini Analyzer 초기화 (매 실행 시 수행되나, 분석 결과 데이터만 캐싱됨)
    gemini_analyzer = None
    if gemini_key and GEMINI_AVAILABLE:
        try:
            # analyzer.py 내의 클래스 사용
            gemini_analyzer = GeminiAnalyzer(gemini_key)
        except Exception as e:
            logging.error(f"Gemini Analyzer 초기화 실패: {e}")
            # UI에 오류를 표시하기 위해 예외를 발생시키지 않고 진행

    if not nara_client and not dapa_client: return {}
    
    # AI 분석기를 포함하여 run_analysis 호출
    results = run_analysis(nara_client, dapa_client, gemini_analyzer, start_date, end_date, min_score)
    return results

# 공통 데이터프레임 표시 설정
def display_dataframe(df, include_winner=False):
    # AI 분석 결과 컬럼은 기본 표시에서 제외 (AI 탭에서 별도 표시)
    base_cols = ['Score', 'Title', 'Source', 'Agency', 'Budget', 'Date', 'Type', 'Details', 'Link', 'MatchedKeywords']
    
    if include_winner:
        base_cols.insert(4, 'Winner')

    # 존재하는 컬럼만 표시
    cols_to_show = [col for col in base_cols if col in df.columns]
    
    # 날짜 형식 통일 시도
    df_display = df.copy()
    if 'Date' in df_display.columns:
        try:
            df_display['Date'] = pd.to_datetime(df_display['Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass

    st.dataframe(
        df_display[cols_to_show],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("링크/규격서", display_text="바로가기"),
            "Score": st.column_config.NumberColumn("점수", format="%d ⭐"),
            "Budget": st.column_config.NumberColumn("금액/예산 (원)", format="%,d"),
            "Date": st.column_config.TextColumn("일시"),
        }
    )

# ============================================
# 사이드바 (설정 및 실행)
# ============================================

with st.sidebar:
    st.header("🔑 API 설정")
    st.info("공공데이터 API 키는 URL 인코딩된 상태로 입력해야 합니다.")
    st.session_state.api_key_nara = st.text_input("나라장터 API 키", value=st.session_state.api_key_nara, type="password")
    st.session_state.api_key_dapa = st.text_input("방위사업청 API 키", value=st.session_state.api_key_dapa, type="password")
    
    # Gemini 키 입력란 추가
    st.session_state.api_key_gemini = st.text_input("Gemini API 키 (AI 분석용)", value=st.session_state.api_key_gemini, type="password")
    if not GEMINI_AVAILABLE:
        st.warning("AI 기능 비활성화됨: google-generativeai 설치 필요")


    st.header("📅 분석 기간 설정")
    today = date.today()
    default_start = today - timedelta(days=30)

    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("시작일", default_start)
    end_date = col_end.date_input("종료일", today)

    st.header("⚙️ 분석 옵션")
    st.session_state.min_score = st.slider("최소 관심 점수 (키워드 기반)", 0, 200, st.session_state.min_score)

    if st.button("🚀 통합 분석 실행", use_container_width=True):
        if not st.session_state.api_key_nara and not st.session_state.api_key_dapa:
            st.error("최소 하나 이상의 조달 API 키를 입력해야 합니다.")
        else:
            # 나라장터 1개월 제약 조건 안내
            if (end_date - start_date).days > 31 and st.session_state.api_key_nara:
                st.info("⚠️ 기간이 1개월 이상일 경우, 나라장터 공고/계약 API 제약으로 월 단위 분할 조회가 실행됩니다.")

            spinner_text = "데이터 수집 및 분석 중..."
            if st.session_state.api_key_gemini and GEMINI_AVAILABLE:
                spinner_text = "데이터 수집 및 AI 심층 분석 중... (시간이 소요될 수 있습니다)"
            
            with st.spinner(spinner_text):
                try:
                    st.session_state.analysis_results = run_cached_analysis(
                        st.session_state.api_key_nara,
                        st.session_state.api_key_dapa,
                        st.session_state.api_key_gemini, # Gemini 키 전달
                        start_date,
                        end_date,
                        st.session_state.min_score
                    )
                    st.success("통합 분석 완료!")
                except Exception as e:
                    st.error(f"분석 실행 중 오류 발생: {e}")
                    logging.exception("Analysis execution error")

# ============================================
# 메인 대시보드
# ============================================

st.markdown('<div class="main-title">🎯 조달 정보 통합 분석 시스템 v12.0 (AI)</div>', unsafe_allow_html=True)

if st.session_state.get('analysis_results'):
    results = st.session_state.analysis_results

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
    if total_count > 0:
        # 결과 데이터프레임 중 하나라도 AI 컬럼이 있는지 확인
        for df in results.values():
            if not df.empty and 'AI_Analysis' in df.columns:
                is_ai_analyzed = True
                break

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌟 총 결과", f"{total_count} 건")
    col2.metric("📋 1. 계획 단계", f"{counts['Plans']} 건")
    col3.metric("📑 2. 규격 단계", f"{counts['Priors']} 건")
    col4.metric("📢 3. 공고 단계", f"{counts['Bids']} 건")
    col5.metric("🤝 4. 계약 단계", f"{counts['Contracts']} 건")

    st.markdown("---")

    # 탭 구조 변경 (AI 분석 탭 동적 추가)
    tabs = ["📋 1. 계획", "📑 2. 규격", "📢 3. 공고", "🤝 4. 계약", "📈 통계"]
    if is_ai_analyzed:
        tabs.insert(0, "🤖 AI 심층 분석")

    tab_contents = st.tabs(tabs)
    tab_index = 0

    # 0. AI 심층 분석 탭
    if is_ai_analyzed:
        with tab_contents[tab_index]:
            st.subheader("🤖 AI 기반 사업 적합성 분석 및 전략 제안")
            st.info("키워드 점수가 높은 순서대로 AI가 회사 프로필과 연계하여 분석한 결과입니다.")

            # 모든 단계의 데이터를 통합하여 점수순 정렬
            all_analyzed_data = pd.concat([df for df in results.values() if not df.empty], ignore_index=True)
            all_analyzed_data = all_analyzed_data.sort_values(by='Score', ascending=False)

            # AI 분석 결과가 정상인 항목만 필터링
            df_ai_results = all_analyzed_data[
                pd.notna(all_analyzed_data.get('AI_Analysis')) & 
                ~all_analyzed_data['AI_Analysis'].str.contains("N/A|오류|미작동|실패", na=False)
            ]

            if df_ai_results.empty:
                st.warning("표시할 AI 분석 결과가 없습니다. (AI 미실행, 분석 실패 또는 관련 항목 없음)")
            else:
                for index, row in df_ai_results.iterrows():
                    st.markdown(f'<div class="ai-card">', unsafe_allow_html=True)
                    # 카드 제목 구성
                    st.markdown(f'<div class="ai-card-title">[{row["Source"]}] {row["Title"]} (점수: {row["Score"]})</div>', unsafe_allow_html=True)
                    
                    col_ai1, col_ai2 = st.columns(2)
                    with col_ai1:
                        st.markdown("**🔍 분석 요약 (Analysis Summary)**")
                        st.write(row['AI_Analysis'])
                    with col_ai2:
                        st.markdown("**💡 전략 제안 (Strategy Proposal)**")
                        st.write(row.get('AI_Strategy', 'N/A'))
                    
                    with st.expander("원본 정보 및 링크"):
                        st.text(f"유형: {row['Type']}, 기관: {row['Agency']}, 예산/금액: {row['Budget']:,.0f} 원")
                        st.markdown(f"[링크 바로가기]({row['Link']})", unsafe_allow_html=True)
                        st.text(f"상세 내용:\n{row['Details']}")

                    st.markdown('</div>', unsafe_allow_html=True)
        tab_index += 1

    # 1. 계획 단계 탭
    with tab_contents[tab_index]:
        st.subheader("📋 발주 및 조달 계획 상세")
        df_plan = results.get('OrderPlans')
        if df_plan is not None and not df_plan.empty:
            display_dataframe(df_plan)
        else:
            st.warning("관련 계획 정보가 없습니다.")
    tab_index += 1

    # 2. 규격 단계 탭
    with tab_contents[tab_index]:
        st.subheader("📑 사전 규격 상세")
        df_prior = results.get('PriorStandards')
        if df_prior is not None and not df_prior.empty:
            display_dataframe(df_prior)
        else:
            st.warning("관련 사전규격 정보가 없습니다.")
    tab_index += 1

    # 3. 공고 단계 탭
    with tab_contents[tab_index]:
        st.subheader("📢 입찰 공고 상세")
        df_bid = results.get('BidNotices')
        if df_bid is not None and not df_bid.empty:
            display_dataframe(df_bid)
        else:
            st.warning("관련 입찰공고 정보가 없습니다.")
    tab_index += 1

    # 4. 계약 단계 탭
    with tab_contents[tab_index]:
        st.subheader("🤝 계약 현황 상세")
        df_contract = results.get('Contracts')
        if df_contract is not None and not df_contract.empty:
             display_dataframe(df_contract, include_winner=True)
        else:
            st.warning("관련 계약 정보가 없습니다.")
    tab_index += 1

    # 5. 통계/시각화 탭
    with tab_contents[tab_index]:
        st.subheader("📈 통합 통계 및 시각화")
        all_data_list = [results.get(key) for key in results if results.get(key) is not None and not results.get(key).empty]
        
        if all_data_list:
            all_data = pd.concat(all_data_list, ignore_index=True)
            
            col_stat1, col_stat2 = st.columns(2)

            with col_stat1:
                # 출처별 분포
                st.plotly_chart(px.pie(all_data, names='Source', title='출처별 데이터 분포', hole=0.3, color_discrete_map={'나라장터':'#1f77b4', '방위사업청':'#ff7f0e'}), use_container_width=True)
            
            with col_stat2:
                # 조달 단계별 분포 시각화
                def categorize_stage(type_str):
                    if "계획" in type_str: return "1. 계획"
                    if "사전규격" in type_str: return "2. 규격"
                    if "입찰공고" in type_str: return "3. 공고"
                    if "계약현황" in type_str: return "4. 계약"
                    return "기타"
                
                all_data['Stage'] = all_data['Type'].apply(categorize_stage)
                
                fig = px.histogram(all_data, x='Stage', title='조달 단계별 건수', color='Source', barmode='group', 
                                   category_orders={"Stage": ["1. 계획", "2. 규격", "3. 공고", "4. 계약"]},
                                   color_discrete_map={'나라장터':'#1f77b4', '방위사업청':'#ff7f0e'})
                st.plotly_chart(fig, use_container_width=True)
            
            # 세부 유형별 분포
            fig_type = px.histogram(all_data, x='Type', title='세부 유형별 건수', color='Source', color_discrete_map={'나라장터':'#1f77b4', '방위사업청':'#ff7f0e'})
            fig_type.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_type, use_container_width=True)

        else:
            st.info("시각화할 데이터가 없습니다.")

else:
    st.info("⬅️ 사이드바에서 설정을 완료하고 '통합 분석 실행' 버튼을 눌러주세요.")
    st.success("✅ 시스템 준비 완료: 나라장터 및 방위사업청의 전 주기 정보(계획, 규격, 공고, 계약)와 AI 심층 분석이 가능합니다.")