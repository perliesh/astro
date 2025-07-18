import streamlit as st
import numpy as np
import plotly.graph_objs as go
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body, solar_system_ephemeris, get_sun, EarthLocation
from astropy.time import Time
from datetime import datetime
import astropy.units as u
import matplotlib.pyplot as plt

st.set_page_config(page_title="천체 좌표 시뮬레이터", layout="wide")
st.title("🔭 천체 좌표 시뮬레이터")
st.markdown("FITS 파일을 업로드하면 해당 천체의 위치를 지평 좌표계와 적도 좌표계를 이용해 시각화 및 시뮬레이션합니다")

# --- 서울 위치 (고정) ---
seoul_location = EarthLocation(lat=37.5665*u.deg, lon=126.9780*u.deg, height=50*u.m)

# --- FITS 파일 업로드 ---
uploaded_file = st.file_uploader("FITS 파일을 업로드하세요(WCS 또는 RA/DEC 포함)", type=["fits", "fit", "fz"])

if uploaded_file:
    try:
        with fits.open(uploaded_file) as hdul:
            # 이미지 데이터가 있는 첫 HDU 찾기 (2차원 이상 이미지 데이터만)
            image_hdu = None
            for hdu in hdul:
                if hdu.is_image and hdu.data is not None and hdu.data.ndim >= 2:
                    image_hdu = hdu
                    break

            if image_hdu is None:
                st.error("FITS 파일에 이미지 데이터가 포함된 HDU를 찾을 수 없습니다.")
                st.stop()
            
            header = image_hdu.header
            data = image_hdu.data
            data = np.nan_to_num(data)

            st.success(f"**'{uploaded_file.name}'** 파일을 성공적으로 처리했습니다.")
            
            # --- WCS 정보 추출 시도 ---
            ra, dec = None, None
            wcs = None
            try:
                wcs = WCS(header)
                ny, nx = data.shape[-2], data.shape[-1]  # 2D 이미지 크기
                x_center, y_center = (nx-1) / 2.0, (ny-1) / 2.0
                skycoord_center = wcs.pixel_to_world(x_center, y_center)
                ra = skycoord_center.ra.deg
                dec = skycoord_center.dec.deg
                st.success(f"이미지 중심 좌표 (WCS 기준): RA={ra:.5f}°, DEC={dec:.5f}°")
            except Exception as e:
                st.warning(f"WCS 해석 실패: {e}")
                wcs = None

                # WCS 해석 실패 시 헤더에서 RA/DEC 직접 추출 시도
                ra = None
                dec = None
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

            # 현재 시간 기준
            now = Time(datetime.utcnow(), scale='utc')

            # 천체 SkyCoord 생성
            try:
                if isinstance(ra, (str, int, float)) or isinstance(dec, (str, int, float)):
                    skycoord = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg), frame='icrs')
                    ra = skycoord.ra.deg
                    dec = skycoord.dec.deg
                else:
                    skycoord = SkyCoord(ra = ra*u.deg, dec = dec*u.deg, frame = 'icrs')
            except Exception as e:
                st.error(f"Skycoord 변환 실패: {e}")
                st.stop()

            altaz_frame = AltAz(obstime=now, location=seoul_location)
            altaz_now = skycoord.transform_to(altaz_frame)
            altitude = altaz_now.alt.deg
            azimuth = altaz_now.az.deg

            #태양 및 달 위치 계산
            with solar_system_ephemeris.set('builtin'):
                sun_coord = get_body('sun', now)
                moon_coord = get_body('moon', now, location=seoul_location)

            sun_altaz = sun_coord.transform_to(altaz_frame)
            moon_altaz = moon_coord.transform_to(altaz_frame)
            sun_alt = sun_altaz.alt.deg
            moon_alt = moon_altaz.alt.deg
            
            st.markdown("지평 좌표계와 적도 좌표계 중 하나를 선택해, 해당 좌표계에서의 천체 위치를 시각화하고 시뮬레이션해볼 수 있습니다. 좌표계를 이용한 위치 정보를 바탕으로 해당 천체의 관측 가능성도 확인해보세요.")
            tabs = st.tabs(["지평 좌표계", "적도 좌표계", "관측 가능성"])
            with tabs[0]:
                
            #지평 좌표계
                st.subheader("지평 좌표계: 시간에 따른 고도 및 방위각 변화 시뮬레이션")
                hours_to_simulate = st.slider("시뮬레이션 시간: 몇 시간 동안 시뮬레이션 할까요?", 1, 24, 6)
                time_steps = 100
                times = now + np.linspace(0, hours_to_simulate, time_steps)*u.hour
                altazs = skycoord.transform_to(AltAz(obstime=times, location=seoul_location))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times.datetime, y=altazs.alt.deg, mode='lines+markers', name='고도 (°)'))
                fig.add_trace(go.Scatter(x=times.datetime, y=altazs.az.deg, mode='lines+markers', name='방위각 (°)'))
                fig.update_layout(
                    title="지평 좌표계: 시간에 따른 고도 및 방위각 변화",
                    xaxis_title="시간 (UTC)",
                    yaxis_title="각도 (°)",
                    height=500,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]: 
                #적도 좌표계
                st.subheader("적도 좌표계: 황도선과 FITS 천체 그래프")
                st.markdown(f"적경: {ra:.5f}, 적위: {dec:.5f}")
                ra_vals = np.linspace(0,360,1000)
                dec_vals = 23.44*np.sin(np.radians(ra_vals))

                fig_ecliptic, ax = plt.subplots(figsize=(10,5))
                ax.plot(ra_vals, dec_vals, label='Ecliptic line', color='orange', linewidth=1.5)
                ax.scatter(ra, dec, color='red', label='FITS cellestial body', s=50, zorder=3)
                ax.set_xlim(0, 360)
                ax.set_ylim(-90, 90)
                ax.set_xlabel('Right Ascension (deg)')
                ax.set_ylabel('Declination (deg)')
                ax.set_title('Equatorial Coordinate System with Ecliptic Line')
                ax.grid(True)
                ax.legend()

                st.pyplot(fig_ecliptic)

                st.subheader("적도 좌표계: 천구에서의 천체 위치")

                with solar_system_ephemeris.set('builtin'):
                    sun_coord = get_body('sun', now, location=seoul_location)
                    moon_coord = get_body('moon', now, location=seoul_location)
                def spherical_to_cartesian(ra_deg, dec_deg):
                    ra_rad = np.radians(ra_deg)
                    dec_rad = np.radians(dec_deg)
                    x = np.cos(dec_rad)*np.cos(ra_rad)
                    y = np.cos(dec_rad)*np.sin(ra_rad)
                    z = np.sin(dec_rad)
                    return x, y, z
                
                obj_x, obj_y, obj_z = spherical_to_cartesian(ra, dec)
                sun_x, sun_y, sun_z = spherical_to_cartesian(sun_coord.ra.deg, sun_coord.dec.deg)
                moon_x, moon_y, moon_z = spherical_to_cartesian(moon_coord.ra.deg, moon_coord.dec.deg)
                earth_x, earth_y, earth_z = spherical_to_cartesian(0,0)

                fig3d = go.Figure()

                sphere_u = np.linspace(0, 2*np.pi, 100)
                sphere_v = np.linspace(0, np.pi, 50)
                sphere_x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
                sphere_y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
                sphere_z = np.outer(np.ones(np.size(sphere_u)), np.cos(sphere_v))

                fig3d.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z, opacity=0.2, colorscale='Blues', showscale=False))

                fig3d.add_trace(go.Scatter3d(x=[earth_x], y=[earth_y], z=[earth_z],
                                            mode='markers+text',
                                            marker=dict(size=7, color='green'),
                                            text=['Earth'], textposition='top center'))
                fig3d.add_trace(go.Scatter3d(x=[sun_x], y=[sun_y], z=[sun_z],
                                            mode='markers+text',
                                            marker=dict(size=10, color='yellow'),
                                            text=['Sun'], textposition='top center'))
                fig3d.add_trace(go.Scatter3d(x=[moon_x], y=[moon_y], z=[moon_z],
                                            mode='markers+text',
                                            marker=dict(size=8, color='gray'),
                                            text=['Moon'], textposition='top center'))
                fig3d.add_trace(go.Scatter3d(x=[obj_x], y=[obj_y], z=[obj_z],
                                            mode='markers+text',
                                            marker=dict(size=6, color='red'),
                                            text=['FITS Object'], textposition='top center'))
                
                north_pole = spherical_to_cartesian(0, 90)
                south_pole = spherical_to_cartesian(0, -90)

                fig3d.add_trace(go.Scatter3d(x=[north_pole[0]], y=[north_pole[1]], z=[north_pole[2]],
                                            mode='markers+text',
                                            marker=dict(size=6, color='blue'),
                                            text=['North Celestial Pole'], textposition='bottom center'))
                fig3d.add_trace(go.Scatter3d(x=[south_pole[0]], y=[south_pole[1]], z=[south_pole[2]],
                                            mode='markers+text',
                                            marker=dict(size=6, color='blue'),
                                            text=['South Celestial Pole'], textposition='top center'))

                equator_ra = np.linspace(0, 360, 360)
                equator_x, equator_y, equator_z = [], [], []
                for ra in equator_ra:
                    x, y, z = spherical_to_cartesian(ra, 0)
                    equator_x.append(x)
                    equator_y.append(y)
                    equator_z.append(z)
                fig3d.add_trace(go.Scatter3d(x=equator_x, y=equator_y, z=equator_z,
                                            mode='lines',
                                            line=dict(color='green', width=2),
                                            name='Celestial Equator'))

                ecliptic_x, ecliptic_y, ecliptic_z = [], [], []
                for ra in equator_ra:
                    dec = 23.5 * np.sin(np.radians(ra))  # 황도면 경사
                    x, y, z = spherical_to_cartesian(ra, dec)
                    ecliptic_x.append(x)
                    ecliptic_y.append(y)
                    ecliptic_z.append(z)
                fig3d.add_trace(go.Scatter3d(x=ecliptic_x, y=ecliptic_y, z=ecliptic_z,
                                            mode='lines',
                                            line=dict(color='orange', width=2),
                                            name='Ecliptic'))

                seasonal_points = {
                    'Vernal Equinox (춘분점)': (0, 0),
                    'Autumnal Equinox (추분점)': (180, 0),
                    'Summer Solstice (하지점)': (90, 23.5),
                    'Winter Solstice (동지점)': (270, -23.5)
                }

                for label, (ra, dec) in seasonal_points.items():
                    x, y, z = spherical_to_cartesian(ra, dec)
                    fig3d.add_trace(go.Scatter3d(x=[x], y=[y], z=[z],
                                                mode='markers+text',
                                                marker=dict(size=6, color='purple'),
                                                text=[label], textposition='top center'))

                fig3d.update_layout(scene=dict(
                    xaxis=dict(title='X (RA)', showbackground=False, showticklabels=False, zeroline=False),
                    yaxis=dict(title='Y (RA)', showbackground=False, showticklabels=False, zeroline=False),
                    zaxis=dict(title='Z (DEC)', showbackground=False, showticklabels=False, zeroline=False),
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=600,
                title='3D Celestial Sphere Visualization')

                st.plotly_chart(fig3d, use_container_width=True)     

            with tabs[2]:
                def compute_observability(target_coord, obs_time, observer_location):
                    # 천체의 지평 좌표 계산
                    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
                    target_altaz = target_coord.transform_to(altaz_frame)
                    target_alt = target_altaz.alt.deg
                    target_az = target_altaz.az.deg

                    # 해와 달 정보
                    sun_coord = get_sun(obs_time).transform_to(altaz_frame)
                    moon_coord = get_body('moon', obs_time, location=observer_location).transform_to(altaz_frame)

                    sun_alt = sun_coord.alt.deg
                    moon_alt = moon_coord.alt.deg
                    moon_angle = target_coord.separation(get_body('moon', obs_time, location=observer_location)).deg

                    # --- 관측 가능성 평가 요소들 ---

                    # 1. 고도 점수: 고도가 높을수록 대기 영향 적음
                    if target_alt > 45:
                        score_alt = 1.0
                    elif target_alt > 20:
                        score_alt = 0.7
                    elif target_alt > 10:
                        score_alt = 0.3
                    else:
                        score_alt = 0.0

                    # 2. 일몰 상태 점수: 해가 지평선 아래에 있어야 어두움
                    if sun_alt < -18:
                        score_sun = 1.0  # 천문박명 이후 (매우 어두움)
                    elif sun_alt < -6:
                        score_sun = 0.5  # 시민박명~천문박명 사이
                    else:
                        score_sun = 0.0  # 밝음 (낮)

                    # 3. 달 영향 점수: 달이 아래 있거나 멀리 떨어져 있을수록 유리
                    if moon_alt < 0:
                        score_moon = 1.0  # 달이 아래
                    elif moon_angle > 60:
                        score_moon = 0.8
                    elif moon_angle > 30:
                        score_moon = 0.4
                    else:
                        score_moon = 0.2

                    # 4. Airmass 점수 (대기질량): 고도 기준 보완
                    zenith_angle = 90 - target_alt
                    airmass = 1 / np.cos(np.radians(zenith_angle)) if target_alt > 0 else np.inf
                    score_airmass = np.exp(-0.2 * (airmass - 1)) if airmass < 3 else 0.0

                    # --- 최종 점수 계산 ---
                    total_score = (score_alt + score_sun + score_moon + score_airmass) / 4

                    return {
                        "target_alt": target_alt,
                        "target_az": target_az,
                        "sun_alt": sun_alt,
                        "moon_alt": moon_alt,
                        "moon_angle": moon_angle,
                        "airmass": airmass,
                        "observability_score": round(total_score, 2)
                    } 
                                # --- 관측 가능성 계산 및 출력 ---
                result = compute_observability(skycoord, now, seoul_location)

                st.subheader("관측 가능성 분석 결과")

                st.markdown(f"""
                    - 🌍 **현재 고도 (Altitude)**: {result['target_alt']:.2f}°
                    - 🧭 **방위각 (Azimuth)**: {result['target_az']:.2f}°
                    - ☀️ **태양 고도**: {result['sun_alt']:.2f}°
                    - 🌙 **달 고도**: {result['moon_alt']:.2f}°
                    - 🌕 **달과의 각거리**: {result['moon_angle']:.2f}°
                    - 🌫️ **대기질량 (Airmass)**: {result['airmass']:.2f}
                    - 🔭 **관측 가능성 점수 (0.0–1.0)**: **{result['observability_score']}**
                """)
                st.markdown("관측 가능성은 현재 고도, 태양 고도, 달 고도, 달과의 거리, 대기질량을 모두 고려해 계산됩니다.")

                # 관측 등급 해석
                score = result["observability_score"]
                if score >= 0.8:
                    st.success("✅ 이 천체는 **매우 관측하기 좋은 조건**입니다.")
                elif score >= 0.5:
                    st.info("⚠️ 이 천체는 **관측이 가능하지만 조건이 완벽하진 않습니다.**")
                else:
                    st.warning("❌ 이 천체는 **관측하기 어려운 조건**입니다.")


    except Exception as e:
        st.error(f"FITS 파일 처리 중 오류 발생: {e}")

else:
    st.info("시작하려면 FITS 파일을 업로드하세요.")
