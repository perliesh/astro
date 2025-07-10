import streamlit as st

import numpy as np

from astropy.io import fits

from PIL import Image

from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from astropy.time import Time

from datetime import datetime


# --- Streamlit 앱 페이지 설정 ---

st.set_page_config(page_title="천문 이미지 분석기", layout="wide")

st.title("🔭 천문 이미지 처리 앱")


# --- 파일 업로더 ---

uploaded_file = st.file_uploader(

    "분석할 FITS 파일을 선택하세요.",

    type=['fits', 'fit', 'fz']

)


# --- 서울 위치 설정 (고정값) ---

seoul_location = EarthLocation(lat=37.5665, lon=126.9780, height=50)  # 서울 위도/경도/고도


# --- 현재 시간 (UTC 기준) ---

now = datetime.utcnow()

now_astropy = Time(now)


# --- 파일이 업로드되면 실행될 로직 ---

if uploaded_file:

    try:

        with fits.open(uploaded_file) as hdul:

            image_hdu = None

            for hdu in hdul:

                if hdu.data is not None and hdu.is_image:

                    image_hdu = hdu

                    break


            if image_hdu is None:

                st.error("파일에서 유효한 이미지 데이터를 찾을 수 없습니다.")

            else:

                header = image_hdu.header

                data = image_hdu.data

                data = np.nan_to_num(data)


                st.success(f"**'{uploaded_file.name}'** 파일을 성공적으로 처리했습니다.")

                col1, col2 = st.columns(2)


                with col1:

                    st.header("이미지 정보")

                    st.text(f"크기: {data.shape[1]} x {data.shape[0]} 픽셀")

                    if 'OBJECT' in header:

                        st.text(f"관측 대상: {header['OBJECT']}")

                    if 'EXPTIME' in header:

                        st.text(f"노출 시간: {header['EXPTIME']} 초")


                    st.header("물리량")

                    mean_brightness = np.mean(data)

                    st.metric(label="이미지 전체 평균 밝기", value=f"{mean_brightness:.2f}")


                with col2:

                    st.header("이미지 미리보기")

                    if data.max() == data.min():

                        norm_data = np.zeros(data.shape, dtype=np.uint8)

                    else:

                        scale_min = np.percentile(data, 5)

                        scale_max = np.percentile(data, 99.5)

                        data_clipped = np.clip(data, scale_min, scale_max)

                        norm_data = (255 * (data_clipped - scale_min) / (scale_max - scale_min)).astype(np.uint8)


                    img = Image.fromarray(norm_data)

                    st.image(img, caption="업로드된 FITS 이미지", use_container_width=True)



                # --- 사이드바: 현재 천체 위치 계산 ---

                st.sidebar.header("🧭 현재 천체 위치 (서울 기준)")


                if 'RA' in header and 'DEC' in header:

                    try:

                        target_coord = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=('hourangle', 'deg'))

                        altaz = target_coord.transform_to(AltAz(obstime=now_astropy, location=seoul_location))

                        altitude = altaz.alt.degree

                        azimuth = altaz.az.degree


                        st.sidebar.metric("고도 (°)", f"{altitude:.2f}")

                        st.sidebar.metric("방위각 (°)", f"{azimuth:.2f}")

                    except Exception as e:

                        st.sidebar.warning(f"천체 위치 계산 실패: {e}")

                else:

                    st.sidebar.info("FITS 헤더에 RA/DEC 정보가 없습니다.")


    except Exception as e:

        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

        st.warning("파일이 손상되었거나 유효한 FITS 형식이 아닐 수 있습니다.")

else:

    st.info("시작하려면 FITS 파일을 업로드해주세요.")


# --- 💬 댓글 기능 (세션 기반) ---

st.divider()

st.header("💬 의견 남기기")


if "comments" not in st.session_state:

    st.session_state.comments = []


with st.form(key="comment_form"):

    name = st.text_input("이름을 입력하세요", key="name_input")

    comment = st.text_area("댓글을 입력하세요", key="comment_input")

    submitted = st.form_submit_button("댓글 남기기")


    if submitted:

        if name.strip() and comment.strip():

            st.session_state.comments.append((name.strip(), comment.strip()))

            st.success("댓글이 저장되었습니다.")

        else:

            st.warning("이름과 댓글을 모두 입력해주세요.")


if st.session_state.comments:

    st.subheader("📋 전체 댓글")

    for i, (n, c) in enumerate(reversed(st.session_state.comments), 1):

        st.markdown(f"**{i}. {n}**: {c}")

else:

    st.info("아직 댓글이 없습니다. 첫 댓글을 남겨보세요!")