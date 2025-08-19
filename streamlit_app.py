import streamlit as st
import analyzer
import pandas as pd
from datetime import datetime, timedelta # [추가] 날짜 계산을 위해 필요

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

# --- 메인 페이지 UI (전면 수정) ---
st.subheader("분석 설정")

# 3개의 컬럼으로 레이아웃 재구성
col_date, col_type, col_keywords = st.columns([1, 1, 2])

with col_date:
    # [신규] 검색 기간 설정 UI
    st.markdown("📅 **검색 기간 설정**")
    # 기본값 설정: 오늘로부터 90일 전 ~ 오늘
    today = datetime.now().date()
    default_start_date = today - timedelta(days=90)
    
    # 날짜 범위 입력 위젯
    date_range = st.date_input(
        "기간 선택 (시작일, 종료일)",
        (default_start_date, today),
        max_value=today,
        format="YYYY-MM-DD",
        label_visibility="visible",
        help="선택한 기간 동안의 사전규격, 입찰공고, 계약 정보를 조회하며, 해당 기간에 포함된 연도의 발주계획을 조회합니다."
    )
    
    # 입력된 날짜 처리
    if date_range and len(date_range) == 2:
        input_start_date, input_end_date = date_range
    elif date_range and len(date_range) == 1:
        # 사용자가 날짜를 하나만 선택한 경우
        input_start_date, input_end_date = date_range[0], date_range[0]
    else:
        # 기본값 사용
        input_start_date, input_end_date = default_start_date, today

with col_type:
    st.markdown("💡 **분석 유형 선택**")
    search_type = st.radio("유형 선택", ('전체 키워드 (AI 확장)', '포커스 키워드'), label_visibility="collapsed", help="전체 키워드 분석 시 Gemini를 활용하여 자동으로 키워드를 확장하고 저장합니다.")

with col_keywords:
    st.markdown("🎯 **포커스 키워드 입력**")
    # 높이를 조절하여 다른 컬럼과 맞춤
    focus_keywords_str = st.text_area("포커스 키워드 (쉼표로 구분)", placeholder="예: 드론, CQB, 가상현실", disabled=(search_type == '전체 키워드 (AI 확장)'), height=110)

# 분석 시작 버튼
if st.button("분석 시작", type="primary", use_container_width=True):
    if not service_key or not gemini_key:
        st.error("오류: 사이드바에서 공공데이터 서비스 키와 Gemini API 키를 모두 입력해야 합니다.")
    elif input_start_date > input_end_date:
         st.error("오류: 시작일이 종료일보다 늦을 수 없습니다.")
    else:
        try:
            client = analyzer.NaraJangteoApiClient(service_key=service_key)
            analyzer.setup_database()
            
            # [수정] 분석 함수 호출 시 선택된 날짜 전달
            if search_type == '포커스 키워드':
                search_keywords = {k.strip() for k in focus_keywords_str.split(',') if k.strip()}
                if not search_keywords:
                    st.error("오류: 포커스 검색을 선택했지만 키워드를 입력하지 않았습니다.")
                else:
                    with st.spinner(f'[{input_start_date} ~ {input_end_date}] 포커스 키워드로 데이터를 수집하고 분석하는 중입니다...'):
                        # 날짜 인자 전달
                        st.session_state.results = analyzer.run_analysis(search_keywords, client, gemini_key, start_date=input_start_date, end_date=input_end_date, auto_expand_keywords=False)
            else:
                search_keywords = analyzer.load_keywords(analyzer.INITIAL_KEYWORDS)
                with st.spinner(f'[{input_start_date} ~ {input_end_date}] 전체 키워드로 데이터를 수집하고 분석/확장하는 중입니다...'):
                    # 날짜 인자 전달
                    st.session_state.results = analyzer.run_analysis(search_keywords, client, gemini_key, start_date=input_start_date, end_date=input_end_date, auto_expand_keywords=True)
            
            # 결과가 성공적으로 생성되면 페이지 새로고침하여 결과 표시
            st.rerun()
            
        except Exception as e:
            st.error(f"🚨 분석 실행 중 예상치 못한 오류가 발생했습니다: {e}")
            st.exception(e) # 상세 오류 로그 표시

# --- 결과 표시 ---
if st.session_state.results:
    st.markdown("---")
    st.header("📊 분석 결과")
    results = st.session_state.results
    
    # 리스크 현황판 및 다운로드 버튼
    risk_df = results.get("risk_df")
    
    # 보고서 다운로드 버튼 (데이터가 있을 경우에만 표시)
    if results.get("report_file_data"):
        st.download_button(
            label=f"📂 통합 엑셀 보고서 다운로드",
            data=results["report_file_data"],
            file_name=results["report_filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
         st.warning("선택한 기간 및 키워드에 해당하는 조달 데이터가 없어 보고서가 생성되지 않았습니다.")

    if risk_df is not None and not risk_df.empty:
        st.subheader("⚠️ 사업 현황 리스크 분석")
        st.dataframe(risk_df, use_container_width=True)
        st.caption("리스크 등급은 공고 후 경과 시간(분석 시점 기준), 사전규격 공개 여부, 추정 가격 등을 바탕으로 자동 분석됩니다.")

    
    # Gemini 리포트
    if results.get("gemini_report"):
        st.markdown("---")
        st.subheader("✨ Gemini 전략 분석 리포트")
        st.markdown(results["gemini_report"], unsafe_allow_html=True)
        
    # 실행 로그
    with st.expander("📝 실행 로그 보기"):
        st.text_area("", value="\n".join(results.get("log", [])), height=300, key="log_results")