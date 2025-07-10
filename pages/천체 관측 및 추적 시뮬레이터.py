import streamlit as st
import numpy as np
import plotly.graph_objs as go
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from datetime import datetime
import astropy.units as u

st.set_page_config(page_title="천체 위치 시뮬레이터", layout="wide")
st.title("🔭 천체 위치 추적 및 관측 시뮬레이터")

# --- 서울 위치 (고정) ---
seoul_location = EarthLocation(lat=37.5665*u.deg, lon=126.9780*u.deg, height=50)

# --- FITS 파일 업로드 ---
uploaded_file = st.file_uploader("FITS 파일을 업로드하세요 (WCS 또는 RA/DEC 포함)", type=["fits", "fit", "fz"])

if uploaded_file:
    try:
        with fits.open(uploaded_file) as hdul:
            # 이미지 데이터가 있는 첫 HDU 찾기 (2차원 이상 이미지 데이터만)
            image_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.is_image:
                    image_hdu = hdu
                    break

            if image_hdu is None:
                st.error("FITS 파일에 이미지 데이터가 포함된 HDU를 찾을 수 없습니다.")
            else:
                header = image_hdu.reader
                data = image_hdu._data
                data = np.nan_to_num(data)

                st.success(f"**'{uploaded_file.name}'** 파일을 성공적으로 처리했습니다.")

            # header, data 정의
            header = image_hdu.header
            data = image_hdu.data

            # --- WCS 정보 추출 시도 ---
            try:
                wcs = WCS(header)
                ny, nx = data.shape[-2], data.shape[-1]  # 2D 이미지 크기
                x_center, y_center = nx / 2, ny / 2
                skycoord_center = wcs.pixel_to_world(x_center, y_center)
                ra = skycoord_center.ra.deg
                dec = skycoord_center.dec.deg
                st.success(f"이미지 중심 좌표 (WCS 기준): RA={ra:.5f}°, DEC={dec:.5f}°")
            except Exception as e:
                st.warning(f"WCS 해석 실패: {e}")

                # WCS 해석 실패 시 헤더에서 RA/DEC 직접 추출 시도
                # 흔히 'RA', 'DEC' 대신 'OBJCTRA', 'OBJCTDEC' 혹은 'CRVAL1', 'CRVAL2' 쓰이기도 함
                ra = None
                dec = None

                # 후보 키 리스트
                ra_keys = ['RA', 'OBJCTRA', 'CRVAL1']
                dec_keys = ['DEC', 'OBJCTDEC', 'CRVAL2']

                for key in ra_keys:
                    if key in header:
                        ra = header[key]
                        break
                for key in dec_keys:
                    if key in header:
                        dec = header[key]
                        break

                if ra is not None and dec is not None:
                    st.success(f"헤더 RA/DEC 사용: RA={ra}°, DEC={dec}°")
                else:
                    st.error("WCS 정보 및 RA/DEC 헤더가 모두 없습니다.")
                    st.stop()

            # 현재 시간 기준 Alt/Az 계산
            now = Time(datetime.utcnow())
            skycoord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            altaz_now = skycoord.transform_to(AltAz(obstime=now, location=seoul_location))
            altitude = altaz_now.alt.deg
            azimuth = altaz_now.az.deg

            st.markdown(f"### 현재 시간 (UTC): {now.iso}")
            st.markdown(f"**서울 기준 현재 위치 → 고도: {altitude:.2f}°, 방위각: {azimuth:.2f}°**")

            # 관측 가능 여부 간단 판단
            if altitude > 10:
                st.success("이 천체는 현재 서울에서 관측 가능합니다! (고도 > 10°)")
            else:
                st.warning("현재 이 천체는 서울에서 낮거나 지평선 아래에 있습니다.")

            # --- 시간별 Alt/Az 변화 시뮬레이션 ---
            st.subheader("시간에 따른 고도 및 방위각 변화 시뮬레이션")
            hours_to_simulate = st.slider("몇 시간 동안 시뮬레이션 할까요?", 1, 24, 6)
            time_steps = 100
            times = now + np.linspace(0, hours_to_simulate, time_steps)*u.hour
            altazs = skycoord.transform_to(AltAz(obstime=times, location=seoul_location))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times.datetime, y=altazs.alt.deg, mode='lines+markers', name='고도 (°)'))
            fig.add_trace(go.Scatter(x=times.datetime, y=altazs.az.deg, mode='lines+markers', name='방위각 (°)'))
            fig.update_layout(
                title="시간에 따른 고도 및 방위각 변화",
                xaxis_title="시간 (UTC)",
                yaxis_title="각도 (°)",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 천구 지도에 현재 위치 표시 ---
            st.subheader("천구 좌표계 상의 현재 천체 위치")
            fig_map = go.Figure()

            # RA를 0~360이 아닌 -180~180 범위로 변환 (Aitoff 투영 맞춤)
            ra_wrap = (ra + 180) % 360 - 180

            fig_map.add_trace(go.Scattergeo(
                lon=[ra_wrap],
                lat=[dec],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=["천체 위치"],
                textposition="top center",
                name="FITS 중심 천체"
            ))

            fig_map.update_geos(
                projection_type="aitoff",
                showcountries=False, showcoastlines=False, showland=False,
                lonaxis=dict(showgrid=True),
                lataxis=dict(showgrid=True),
                resolution=50,
                lonaxis_range=[-180, 180]
            )

            fig_map.update_layout(
                title="Aitoff 투영: 천구 좌표계",
                height=500,
                margin={"r":0,"t":30,"l":0,"b":0}
            )
            st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"FITS 파일 처리 중 오류 발생: {e}")

else:
    st.info("시작하려면 FITS 파일을 업로드하세요.")
