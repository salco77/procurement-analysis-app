import streamlit as st
import analyzer
import pandas as pd
from datetime import datetime, timedelta
import sys
import importlib

# --- Streamlit 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 조달 정보 분석 시스템",
    page_icon="🚀",
    layout="wide",
)

# --- 세션 상태 초기화 ---
if 'results' not in st.session_state:
    st.session_state.results = None
# AI 점수 임계값 세션 상태 초기화 (기본값 60점)
if 'min_score_threshold' not in st.session_state:
    st.session_state.min_score_threshold = 60

# --- UI 레이아웃 ---
st.title("🚀 AI 기반 조달 정보 탐지 및 분석 시스템")
st.markdown("키워드 기반 탐색과 AI 기반 광범위 탐색을 결합하여 잠재적 사업 기회를 발굴합니다.")
st.markdown("---")

# --- 사이드바 UI ---
with st.sidebar:
    
    # 진단 정보 표시
    st.header("🔍 진단 정보")
    if hasattr(analyzer, 'setup_database'):
        st.success("✅ analyzer.py 로드 정상")
    else:
        st.error("❌ analyzer.py 로드 비정상. 코드를 확인하고 앱을 재시작/재배포하세요.")
        if st.button("모듈 다시 로드 시도 (개발용)"):
            if 'analyzer' in sys.modules:
                importlib.reload(analyzer)
            st.rerun()

    st.markdown("---")

    st.header("🛠️ 설정 및 관리")
    service_key = st.text_input("공공데이터 서비스 키", type="password", placeholder="공공데이터포털 키를 입력하세요")
    gemini_key = st.text_input("Gemini API 키", type="password", placeholder="Google AI Studio 키를 입력하세요 (AI 분석용)")
    
    # AI 관련성 점수 임계값 설정 슬라이더
    st.markdown("---")
    st.subheader("⚙️ AI 분석 설정")
    min_score = st.slider(
        "최소 관련성 점수 임계값 (0-100)",
        min_value=0,
        max_value=100,
        value=st.session_state.min_score_threshold,
        step=5,
        help="설정된 점수 이상의 사업만 최종 보고서에 포함됩니다. 결과가 안 보이면 점수를 낮춰보세요."
    )
    st.session_state.min_score_threshold = min_score

    st.markdown("---")
    with st.expander("상세 키워드 목록 관리", expanded=False):
        st.info("상세 키워드에 매칭되는 사업은 관련성 100점으로 처리됩니다. AI는 이 목록을 기반으로 키워드를 자동 확장합니다.")
        try:
            current_keywords = analyzer.load_keywords(analyzer.INITIAL_KEYWORDS)
            keywords_to_remove = st.multiselect("삭제할 키워드:", sorted(list(current_keywords)))
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
# 결과가 없을 때만 분석 설정 표시
if not st.session_state.results:
    st.subheader("분석 설정")

    # 레이아웃 재구성
    col_date, col_options = st.columns([1, 2])

    with col_date:
        st.markdown("📅 **검색 기간 설정**")
        today = datetime.now().date()
        default_start_date = today - timedelta(days=90)
        
        date_range = st.date_input(
            "기간 선택 (시작일, 종료일)",
            (default_start_date, today),
            max_value=today,
            format="YYYY-MM-DD",
            label_visibility="visible"
        )
        
        if date_range and len(date_range) == 2:
            input_start_date, input_end_date = date_range
        elif date_range and len(date_range) == 1:
            input_start_date, input_end_date = date_range[0], date_range[0]
        else:
            input_start_date, input_end_date = default_start_date, today

    with col_options:
        st.markdown("💡 **분석 옵션**")
        # 분석 유형 선택 대신 키워드 확장 옵션 제공
        auto_expand = st.checkbox("AI 기반 자동 키워드 확장 활성화", value=True)
        st.caption("활성화 시, AI가 관련성 높다고 판단한 사업에서 새로운 키워드를 추출하여 자동으로 저장합니다.")
        

    # 분석 시작 버튼
    if st.button("분석 시작", type="primary", use_container_width=True):
        if not service_key:
            st.error("오류: 공공데이터 서비스 키를 입력해야 합니다.")
        elif input_start_date > input_end_date:
            st.error("오류: 시작일이 종료일보다 늦을 수 없습니다.")
        else:
            if not gemini_key:
                st.warning("Gemini API 키가 입력되지 않았습니다. AI 관련성 분석, 전략 리포트, 키워드 확장은 생략됩니다.")

            try:
                client = analyzer.NaraJangteoApiClient(service_key=service_key)
                
                # setup_database 호출
                analyzer.setup_database()
                
                # 상세 키워드 로드
                search_keywords = analyzer.load_keywords(analyzer.INITIAL_KEYWORDS)
                
                with st.spinner(f'[{input_start_date} ~ {input_end_date}] 하이브리드 탐색 및 AI 분석 중... (AI 분석 포함 시 시간이 소요될 수 있습니다)'):
                    # 분석 함수 호출 시 AI 점수 임계값 전달
                    st.session_state.results = analyzer.run_analysis(
                        search_keywords, client, gemini_key, 
                        start_date=input_start_date, end_date=input_end_date, 
                        auto_expand_keywords=auto_expand, 
                        min_relevance_score=st.session_state.min_score_threshold
                    )
                
                st.rerun()

            except AttributeError as e:
                # AttributeError 발생 시 명확한 안내 제공
                st.error(f"🚨 모듈 로드 오류 발생: {e}")
                st.error("analyzer.py 파일이 최신 버전인지 확인하고 애플리케이션을 재시작/재배포 해주세요. (사이드바의 진단 정보를 확인하세요)")
            except Exception as e:
                st.error(f"🚨 분석 실행 중 예상치 못한 오류가 발생했습니다: {e}")
                st.exception(e)

# --- 결과 표시 (수정됨) ---
if st.session_state.results:
    st.markdown("---")
    st.header("📊 분석 결과")
    results = st.session_state.results
    
    # 실행 로그 우선 표시 (진단용)
    st.subheader("📝 실행 로그 (상세 내역)")
    # 로그 영역을 기본으로 확장하여 표시
    with st.expander("로그 보기 (문제 해결을 위해 상세 내용을 확인하세요)", expanded=True):
         st.text_area("로그 상세", value="\n".join(results.get("log", [])), height=400, key="log_results")
    
    # 분석 재시작 버튼 추가
    if st.button("🔄 새로운 분석 시작하기 (설정 변경)", use_container_width=True):
         st.session_state.results = None
         st.rerun()

    st.markdown("---")

    risk_df = results.get("risk_df")
    
    # 보고서 다운로드 버튼
    if results.get("report_file_data"):
        st.success(f"분석 완료. AI 관련성 점수 {st.session_state.min_score_threshold}점 이상인 사업만 포함되었습니다.")
        st.download_button(
            label=f"📂 통합 엑셀 보고서 다운로드 (AI 평가 포함)",
            data=results["report_file_data"],
            file_name=results["report_filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
         # 결과가 없는 경우 경고 메시지 및 로그 확인 안내 강화
         st.error(f"❌ 보고서가 생성되지 않았습니다. 선택한 기간 및 AI 점수 임계값({st.session_state.min_score_threshold}점)에 해당하는 데이터가 없습니다. 상단의 실행 로그를 확인하고 설정을 조정해주세요.")

    # (리스크 현황판, Gemini 리포트 표시는 기존과 동일하게 유지)
    if risk_df is not None and not risk_df.empty:
        st.subheader("⚠️ 사업 현황 리스크 분석")
        st.dataframe(risk_df, use_container_width=True)
        st.caption("리스크 등급은 공고 후 경과 시간(분석 시점 기준), 사전규격 공개 여부, 추정 가격 등을 바탕으로 자동 분석됩니다.")
        
    if results.get("gemini_report"):
        st.markdown("---")
        st.subheader("✨ Gemini 전략 분석 리포트")
        st.markdown(results["gemini_report"], unsafe_allow_html=True)