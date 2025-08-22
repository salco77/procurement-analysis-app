import streamlit as st
import analyzer
import pandas as pd
from datetime import datetime, timedelta
import sys
import importlib
import io # io 모듈 임포트 추가

# --- Streamlit 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 조달 정보 분석 시스템",
    page_icon="🚀",
    layout="wide",
)

# Streamlit rerun 호환성 처리 함수
def rerun_app():
    try:
         st.rerun()
    except AttributeError:
         # 구버전 Streamlit 호환성
         st.experimental_rerun()

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
    
    # [수정됨] 진단 정보 및 버전 확인 섹션 강화
    st.header("🔍 진단 정보 및 버전 확인")
    if hasattr(analyzer, 'setup_database'):
        st.success("✅ analyzer.py 로드 정상")
        
        # [신규 추가] 네거티브 키워드 개수 표시 (버전 확인용)
        # 이 기능으로 현재 로드된 analyzer.py가 최신 버전인지 확인합니다.
        try:
            neg_keyword_count = len(analyzer.NEGATIVE_KEYWORDS)
            # 최신 버전 기준(v3)으로 50개 이상이어야 함
            if neg_keyword_count > 50:
                st.info(f"💡 로드된 네거티브 키워드: {neg_keyword_count}개 (최신 버전 감지됨)")
            else:
                st.error(f"⚠️ 로드된 네거티브 키워드: {neg_keyword_count}개 (구버전 의심). analyzer.py를 최신으로 교체하고 앱을 완전히 재시작(Ctrl+C 후 재실행)하세요.")
        except Exception as e:
            st.error(f"❌ 네거티브 키워드 정보 로드 실패: {e}")

    else:
        st.error("❌ analyzer.py 로드 비정상. 코드를 확인하고 앱을 재시작/재배포하세요.")

    # [개선됨] 모듈 강제 다시 로드 버튼
    if st.button("🔄 모듈 강제 다시 로드 (실패 시 앱 재시작)"):
        if 'analyzer' in sys.modules:
            try:
                importlib.reload(analyzer)
                st.success("모듈 다시 로드 성공! 페이지를 새로고침합니다.")
            except Exception as e:
                st.error(f"모듈 다시 로드 실패: {e}. 앱을 재시작하세요.")
        rerun_app()

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
            st.markdown(f"**현재 키워드 수: {len(current_keywords)}개**")
            
            # 키워드 목록 보기 (UI 개선)
            if st.checkbox("현재 키워드 목록 보기"):
                st.text_area("키워드 목록", value="\n".join(sorted(list(current_keywords))), height=150, disabled=True)

            keywords_to_remove = st.multiselect("삭제할 키워드:", sorted(list(current_keywords)))
            if st.button("선택한 키워드 삭제"):
                if keywords_to_remove:
                    updated_keywords = current_keywords - set(keywords_to_remove)
                    analyzer.save_keywords(updated_keywords)
                    st.success("키워드가 삭제되었습니다.")
                    rerun_app()
                else:
                    st.warning("삭제할 키워드를 선택해주세요.")
            
            # 텍스트 입력 방식으로 변경 (text_area -> text_input)
            new_keywords_str = st.text_input("추가할 키워드 (쉼표로 구분):")
            if st.button("키워드 추가"):
                if new_keywords_str:
                    new_keywords = {k.strip() for k in new_keywords_str.split(',') if k.strip()}
                    updated_keywords = current_keywords.union(new_keywords)
                    analyzer.save_keywords(updated_keywords)
                    st.success("키워드가 추가되었습니다.")
                    rerun_app()
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
        # 기본 검색 기간을 최근 30일로 단축하여 빠른 분석 유도
        default_start_date = today - timedelta(days=30)
        
        date_range = st.date_input(
            "기간 선택 (시작일, 종료일)",
            (default_start_date, today),
            max_value=today,
            format="YYYY-MM-DD",
            label_visibility="visible"
        )
        
        # 날짜 입력값 처리 로직 개선
        if date_range:
            if len(date_range) == 2:
                input_start_date, input_end_date = date_range
            elif len(date_range) == 1:
                input_start_date, input_end_date = date_range[0], date_range[0]
            else:
                 input_start_date, input_end_date = default_start_date, today
        else:
            input_start_date, input_end_date = default_start_date, today

    with col_options:
        st.markdown("⚙️ **고급 옵션**")
        # AI 키워드 자동 확장 기본값 활성화
        auto_expand = st.checkbox("AI 기반 키워드 자동 확장 사용", value=True, help="분석 결과를 바탕으로 AI가 새로운 키워드를 추천하고 자동으로 목록에 추가합니다.")
        
        # 분석 실행 버튼
        st.markdown("🚀 **분석 실행**")
        if st.button("데이터 수집 및 AI 분석 시작", type="primary"):
            if not service_key:
                st.error("공공데이터 서비스 키는 필수입니다. 사이드바에서 입력해주세요.")
            else:
                # 데이터베이스 초기화
                try:
                    analyzer.setup_database()
                    client = analyzer.NaraJangteoApiClient(service_key)
                    loaded_keywords = analyzer.load_keywords(analyzer.INITIAL_KEYWORDS)

                    # 분석 실행
                    with st.spinner("분석이 진행 중입니다. 데이터 양과 API 상태에 따라 수 분 이상 소요될 수 있습니다..."):
                        # 로그 출력 공간 확보
                        log_placeholder = st.empty()
                        
                        # 실시간 로그 스트리밍 구현 (analyzer.run_analysis 수정 없이 진행)
                        def run_and_stream_log():
                            results = analyzer.run_analysis(
                                search_keywords=loaded_keywords,
                                client=client,
                                gemini_key=gemini_key,
                                start_date=input_start_date,
                                end_date=input_end_date,
                                auto_expand_keywords=auto_expand,
                                min_relevance_score=st.session_state.min_score_threshold
                            )
                            # 최종 로그 표시
                            if results and results.get('log'):
                                log_placeholder.markdown("### 📊 분석 로그\n```\n" + "\n".join(results['log']) + "\n```")
                            return results

                        st.session_state.results = run_and_stream_log()
                        rerun_app()

                except Exception as e:
                    st.error(f"분석 실행 중 치명적 오류 발생: {e}")
                    # 상세 오류 로그 출력
                    import traceback
                    st.code(traceback.format_exc())

# --- 결과 표시 UI ---
if st.session_state.results:
    results = st.session_state.results
    st.success("🎉 분석이 완료되었습니다!")

    # 결과 다운로드 및 새로고침 버튼
    col_dl, col_refresh = st.columns(2)
    with col_dl:
        if results.get("report_file_data"):
            st.download_button(
                label="📥 통합 엑셀 보고서 다운로드",
                data=results["report_file_data"],
                file_name=results["report_filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
    with col_refresh:
        if st.button("🔄 새로운 분석 시작하기"):
            st.session_state.results = None
            rerun_app()

    st.markdown("---")

    # AI 전략 보고서 (가장 먼저 표시)
    if results.get("gemini_report"):
        st.subheader("⭐ AI 맞춤형 전략 분석 보고서 (Gemini)")
        with st.expander("보고서 내용 보기", expanded=True):
            st.markdown(results["gemini_report"])
        st.markdown("---")

    # 탭 기반 결과 표시
    st.subheader("📊 분석 결과 요약")

    tab_main, tab_order_plan, tab_risk, tab_log = st.tabs(["종합 현황 보고서", "발주계획 현황", "리스크 분석", "상세 로그"])

    # 엑셀 파일 데이터를 다시 DataFrame으로 로드 (Streamlit 표시용)
    def load_excel_sheet_to_df(sheet_name):
        if results.get("report_file_data"):
            try:
                # MultiIndex 헤더 처리를 위해 header=[0, 1] 지정
                header_rows = [0, 1] if sheet_name == "종합 현황 보고서" else [0]
                df = pd.read_excel(io.BytesIO(results["report_file_data"]), sheet_name=sheet_name, header=header_rows)
                
                # 빈 셀(NaN)을 빈 문자열로 채우기 (표시 개선)
                df = df.fillna("")
                return df
            except Exception as e:
                st.warning(f"'{sheet_name}' 시트를 로드하는 데 실패했습니다: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    with tab_main:
        st.markdown(f"관련성 점수 **{st.session_state.min_score_threshold}점 이상**의 사업 목록입니다.")
        df_main = load_excel_sheet_to_df("종합 현황 보고서")
        if not df_main.empty:
            # Streamlit 데이터 프레임 사용 (필터링 및 정렬 기능 제공)
            st.dataframe(df_main, use_container_width=True, hide_index=True)
        else:
            st.info("해당하는 데이터가 없습니다. 사이드바에서 최소 관련성 점수 임계값을 낮춰보세요.")

    with tab_order_plan:
        st.markdown("향후 발주 예정인 사업 목록입니다.")
        df_order_plan = load_excel_sheet_to_df("발주계획 현황")
        if not df_order_plan.empty:
             st.dataframe(df_order_plan, use_container_width=True, hide_index=True)
        else:
             st.info("해당하는 발주계획 데이터가 없습니다.")

    with tab_risk:
        st.markdown("진행 중인 사업의 리스크 분석 결과입니다.")
        # 리스크 분석은 analyzer.py에서 직접 DataFrame으로 반환됨
        if results.get("risk_df") is not None and not results["risk_df"].empty:
            st.dataframe(results["risk_df"], use_container_width=True, hide_index=True)
        else:
            st.info("분석 대상 리스크 데이터가 없습니다.")

    with tab_log:
        st.markdown("시스템 실행 상세 로그입니다.")
        if results.get("log"):
            st.code("\n".join(results["log"]))