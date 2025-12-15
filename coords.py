import numpy as np

# WGS-84 椭球参数
a = 6378137.0              # 长半轴
f = 1 / 298.257223563      # 扁率
b = a * (1 - f)            # 短半轴
e2 = 1 - (b*b)/(a*a)       # 第一偏心率平方


def geodetic_to_ecef(lat, lon, h):
    """WGS84 大地坐标系 -> ECEF"""
    lat = np.radians(lat)
    lon = np.radians(lon)
    
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N*(1 - e2) + h) * np.sin(lat)
    return X, Y, Z


def ecef_to_geodetic(X, Y, Z):
    """ECEF -> WGS84 大地坐标(lat, lon, h)"""
    lon = np.arctan2(Y, X)
    p = np.sqrt(X*X + Y*Y)
    
    lat = np.arctan2(Z, p*(1 - e2))  # 初始估计

    # 迭代求解纬度
    for _ in range(30):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p/np.cos(lat) - N
        lat = np.arctan2(Z, p*(1 - e2*(N/(N + h))))

    # 计算最终 N 和 h
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p/np.cos(lat) - N
    
    return np.degrees(lat), np.degrees(lon), h


def enu_to_ecef(e, n, u, lat0, lon0, h0):
    """ENU → ECEF"""
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    # 原点的ECEF
    x0, y0, z0 = geodetic_to_ecef(np.degrees(lat0), np.degrees(lon0), h0)

    # ENU到ECEF的旋转矩阵
    R = np.array([
        [-np.sin(lon0),             np.cos(lon0),              0],
        [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)],
        [ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)]
    ])

    enu_vec = np.array([e, n, u])
    ecef_vec = np.array([x0, y0, z0]) + R.T @ enu_vec
    return ecef_vec


def enu_to_wgs84(e, n, u, lat0, lon0, h0):
    """主函数：ENU → WGS84"""
    X, Y, Z = enu_to_ecef(e, n, u, lat0, lon0, h0)
    lat, lon, h = ecef_to_geodetic(X, Y, Z)
    return lat, lon, h
