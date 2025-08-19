import streamlit as st
import analyzer
import pandas as pd

# --- Streamlit 페이지 기본 설정 ---
st.set_page_config(
    page_title="조달 정보 분석 시스템",
    page_icon="🚀",
    layout="wide",
)

# --- 세션 상태 초기화 ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- UI 레이아웃 ---
st.title("🚀 조달 정보 분석 및 키워드 확장 시스템")
st.markdown("---")

# --- 사이드바 UI ---
with st.sidebar:
    st.header("🛠️ 설정 및 관리")
    service_key = st.text_input("공공데이터 서비스 키", type="password", placeholder="공공데이터포털에서 발급받은 키를 입력하세요")
    gemini_key = st.text_input("Gemini API 키", type="password", placeholder="Google AI Studio에서 발급받은 키를 입력하세요")
    st.markdown("---")
    with st.expander("키워드 목록 관리", expanded=False):
        try:
            current_keywords = analyzer.load_keywords(analyzer.INITIAL_KEYWORDS)
            keywords_to_remove = st.multiselect("삭제할 키워드를 선택하세요:", sorted(list(current_keywords)))
            if st.button("선택한 키워드 삭제"):
                if keywords_to_remove:
                    updated_keywords = current_keywords - set(keywords_to_remove)
                    analyzer.save_keywords(updated_keywords)
                    st.success("키워드가 삭제되었습니다.")
                    st.rerun()
                else:
                    st.warning("삭제할 키워드를 선택해주세요.")
            new_keywords_str = st.text_area("추가할 키워드 (쉼표로 구분):")
            if st.button("키워드 추가"):
                if new_keywords_str:
                    new_keywords = {k.strip() for k in new_keywords_str.split(',') if k.strip()}
                    updated_keywords = current_keywords.union(new_keywords)
                    analyzer.save_keywords(updated_keywords)
                    st.success("키워드가 추가되었습니다.")
                    st.rerun()
                else:
                    st.warning("추가할 키워드를 입력해주세요.")
        except Exception as e:
            st.error(f"키워드 파일 처리 중 오류 발생: {e}")

# --- 메인 페이지 UI ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. 분석 유형 선택")
    search_type = st.radio("분석 유형", ('전체 키워드로 분석 (AI 키워드 확장)', '포커스 키워드로 분석'), label_visibility="collapsed")
with col2:
    st.subheader("2. 포커스 키워드 입력")
    focus_keywords_str = st.text_area("포커스 키워드", placeholder="예: 드론, CQB, 가상현실", disabled=(search_type == '전체 키워드로 분석 (AI 키워드 확장)'))

if st.button("분석 시작", type="primary", use_container_width=True):
    if not service_key or not gemini_key:
        st.error("오류: 사이드바에서 공공데이터 서비스 키와 Gemini API 키를 모두 입력해야 합니다.")
    else:
        try:
            client = analyzer.NaraJangteoApiClient(service_key=service_key)
            analyzer.setup_database()
            if search_type == '포커스 키워드로 분석':
                search_keywords = {k.strip() for k in focus_keywords_str.split(',') if k.strip()}
                if not search_keywords:
                    st.error("오류: 포커스 검색을 선택했지만 키워드를 입력하지 않았습니다.")
                else:
                    with st.spinner('포커스 키워드로 데이터를 수집하고 분석하는 중입니다...'):
                        st.session_state.results = analyzer.run_analysis(search_keywords, client, gemini_key, auto_expand_keywords=False)
            else:
                search_keywords = analyzer.load_keywords(analyzer.INITIAL_KEYWORDS)
                with st.spinner('전체 키워드로 데이터를 수집하고 분석/확장하는 중입니다...'):
                    st.session_state.results = analyzer.run_analysis(search_keywords, client, gemini_key, auto_expand_keywords=True)
            st.rerun()
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- 결과 표시 ---
if st.session_state.results:
    st.markdown("---")
    st.header("📊 분석 결과")
    results = st.session_state.results
    
    # --- [신규] 리스크 현황판 및 다운로드 버튼 ---
    risk_df = results.get("risk_df")
    if risk_df is not None and not risk_df.empty:
        st.subheader("⚠️ 사업 현황 리스크 분석")
        st.dataframe(risk_df, use_container_width=True)
        st.caption("리스크 등급은 공고 후 경과 시간, 사전규격 공개 여부, 추정 가격 등을 바탕으로 자동 분석됩니다.")
    else:
        st.info("현재 분석할 리스크 대상(공고/낙찰 상태) 사업이 없습니다.")

    if results.get("report_file_data"):
        st.download_button(
            label=f"📂 통합 엑셀 보고서 다운로드",
            data=results["report_file_data"],
            file_name=results["report_filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    
    # Gemini 리포트
    if results.get("gemini_report"):
        st.markdown("---")
        st.subheader("✨ Gemini 전략 분석 리포트")
        st.markdown(results["gemini_report"], unsafe_allow_html=True)
        
    # 실행 로그
    with st.expander("📝 실행 로그 보기"):
        st.text_area("", value="\n".join(results.get("log", [])), height=200, key="log_results")