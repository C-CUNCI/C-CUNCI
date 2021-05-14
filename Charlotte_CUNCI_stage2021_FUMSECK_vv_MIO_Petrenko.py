# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:39:21 2021

@author: chacu
"""

from datetime import datetime, timedelta  # Dates in YYYY-MM-DD HH:MM:SS
import matplotlib.pyplot as plt  # plotting package (create plots)
from matplotlib.pyplot import savefig
import numpy as np  # calculations on vectors and matrices, via ndarray
import pandas as pd  # Data analysis in table form
from scipy.io import loadmat  # Load MATLAB file
import warnings  # delete warning: "mean of empty slice" because of nan

import cartopy.crs as ccrs  # Trace the transects on a map
import cartopy.feature as cfeature
from geopy import distance  # Calculate distances



transchoice = [1]  #, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # for each transect
for p in transchoice:
    # Open the MATLAB file Tp.mat with p = 1 to 13
    x = loadmat('T' + str(p) + '.mat')

    lon = x['lon']
    lat = x['lat']
    u = x['u']
    v = x['v']
    w = x['w']
    depth = x['depth'] + 11  # depth begin at 19m and every 8 m : 19 - 8 = 11
    mytime = x['mytime']




# # ==========================================================================
# # A.    Transect length xL, depth yL and time
# # ==========================================================================

    dist = [0]
    n = 0
    for i in range(len(lat)-1):

        lat1 = lat[n]
        lon1 = lon[n]
        point1 = [lat1, lon1]

        lat2 = lat[n+1]
        lon2 = lon[n+1]
        point2 = [lat2, lon2]

        # Geodetic distance between 2 points x1 and x2 in km
        distance_geopy = distance.distance(point1, point2).km

        n = n + 1  # For all the points

        dist.append(distance_geopy)  # Add distances to the list dist
        dist[i+1] = dist[i] + dist[i+1]  # Sum of all distances

    # xL: Number of elements in the transect (494 for transect 1 for example)
    xL = len(dist)  # dist = transect in km; xL is the length of dist
    yL = len(depth)


    # Transect start and end times
    def datenum_to_datetime(datenum):
        """
        Convert Matlab datenum into Python datetime YYYY-MM-DD HH:MM:SS

        :param datenum: Date in datenum format
        :return:        Datetime object corresponding to datenum.
        """
        days = datenum % 1
        hours = days % 1 * 24
        minutes = hours % 1 * 60
        seconds = minutes % 1 * 60
        return datetime.fromordinal(int(datenum)) \
            + timedelta(days=int(days)) \
            + timedelta(hours=int(hours)) \
            + timedelta(minutes=int(minutes)) \
            + timedelta(seconds=int(seconds)) \
            - timedelta(days=366)

    mytime1 = datenum_to_datetime(mytime[0])
    mytime2 = datenum_to_datetime(mytime[xL-1])  # acquisition time: 2min
    # print ("Transect n°" + str(p), "start:", mytime1, "end:", mytime2)
    duration1 = mytime2 - mytime1
    # print ("Transect n°" + str(p), "last", duration1)

    mytime0 = []
    for i in mytime:
        mytime0.append(datenum_to_datetime(i))

    import matplotlib.dates as mdates
    xfmt = mdates.DateFormatter('%H')


# # ==========================================================================
# # B.    Calculation of the average vertical velocity w_average

# # The average value of w over the whole column is representative of the
# # vertical speed of the boat. It is calculated at each point (w_average).
# # ==========================================================================

    w_average = []
    for i in range(xL):
        Σw = w[i]  # Σw = all vertical velocities w at one point
        Σw = Σw[~pd.isnull(Σw)]  # Remove the missing values (nan)
        # Average vertical velocity at this point
        w_average.append(sum(Σw) / len(Σw))


    # # Plot of w_average as a function of distance
    # fig1 = plt.figure(1, figsize=(12, 6))
    # plt.plot(dist, w_average)
    # plt.xlabel("Distance [km]", size=15)
    # plt.ylabel("Vitesse verticale moyenne [m/s]", size=12)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    # plt.title("Variation of the average vertical velocity for "
    #             "the transect n°" + str(p), size=14)
    # # savefig('T'+ str(p) + '_1' + '.png', bbox_inches='tight')
    # # plt.close()



# # ==========================================================================
# # C.    Calculation of the current vertical velocity

# # w_current = calculated velocity w[i] corrected by the
# # average vertical velocity of the boat w_average
# # ==========================================================================


    w_current = []
    for i in range(xL):
        w_current.append(w[i] - w_average[i])

    # Create a transpose matrix wmat_current from the list w_current
    wmat_current = np.transpose(np.asarray(w_current))



# # ==========================================================================
# # D.    MAP + velocity graphs as a function of depth and time
# # ==========================================================================

#     fig2 = plt.figure(2, figsize=(12, 6))
#     gs = fig2.add_gridspec(2, 2)  # Place the subplots according to a 2x2 grid
#     # plt.suptitle("Transect map and velocity graphs for "
#                   #  "the transect n°" + str(p), size=16)

# # # ================
# # # D1.Transect map
# # # ================

#     # 1st graph in position [0, 0], with a Mercator projection
#     ax = fig2.add_subplot(gs[0, 0], projection=ccrs.Mercator())
#     ax.set_title('Localisation du transect n°' + str(p), size=14)
#     plt.figtext(0.28, 0.52, 'Longitude', size=14)
#     plt.figtext(0.12, 0.68, 'Latitude', rotation=90, size=14)

#     ax.set_extent([5.5, 10, 42.5, 44.5])  # coordinates
#     ax.coastlines(resolution='auto', color='k')
#     # resolution 10, 50, 110m or auto

#     A = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
#                       ylocs=[42.66666667, 43, 43.33333333, 43.66666667, 44,
#                             44.33333333], dms=True, linewidth=0.5,
#                       color='grey', alpha=0.5, linestyle='--')
#     # dms minutes, seconds: 20' and 40', (xlocs: the values we want to put)
#     # xlocs=[6, 7, 8, 9, 10],

#     A.top_labels = False
#     A.right_labels = False
#     A.xlabel_style = {'size': 15}  # 'color': 'k'}
#     A.ylabel_style = {'size': 15}


#     # Continents in grey
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m',
#                                                 facecolor='lightgrey'))
#     # Trace the transect in blue
#     plt.scatter(lon, lat, color="dodgerblue", s=2, alpha=0.5,
#                 transform=ccrs.PlateCarree())
#     # Trace the beginning of the transect in a black dot with the number
#     plt.scatter(lon[0], lat[0], color='k',  s=40, alpha=1,
#                 transform=ccrs.PlateCarree())
#     plt.text(lon[0]-0.05, lat[0]-0.2, '' + str(p), size=15, transform=ccrs.PlateCarree())



# # ==============================================================
# # D2. Velocity graphs as a function of depth and time
# # ==============================================================


    
    # np.meshgrid takes 2 1D arrays and produces 2 2D matrices of pairs (x, y)
    time, depth2 = np.meshgrid(mytime0, depth)
    w_currentT = np.transpose(w_current)
    uT = np.transpose(u)
    vT = np.transpose(v)
    wT = np.transpose(w)


    # gs = fig2.add_gridspec(3, 2)  # 3x2 Grid
    # fig2.subplots_adjust(hspace=0.5)  # Space between the graphs
    # fig2.set_size_inches(15, 8)  # Graph size

    # fig2.add_subplot(gs[0, 1])  # u fct depth and time
    # plt.pcolormesh(time, depth2, uT, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(-0.5, 0.5)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=14)
    # plt.title("Vitesse horizontale corrigée u(z,t)", size=14)
    # plt.ylabel("Profondeur [m]", size=14)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    
    # fig2.add_subplot(gs[1, 1])  # v fct depth and time
    # plt.pcolormesh(time, depth2, vT, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(-0.5, 0.5)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=14)
    # plt.title("Vitesse horizontale corrigée v(z,t)", size=14)
    # plt.ylabel("Profondeur [m]", size=14)
    # plt.gca().xaxis.set_major_formatter(xfmt)

    # fig2.add_subplot(gs[2, 1])  # w fct depth and time
    # plt.pcolormesh(time, depth2, wT, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(-0.5, 0.5)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=14)
    # plt.title("Vitesse verticale mesurée $w_{mes}$(z,t)", size=16)
    # plt.xlabel("Temps [h]", size=14)
    # plt.ylabel("Profondeur [m]", size=14)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    
    # fig2.add_subplot(gs[2, 0])  # w_current fct depth and time
    # plt.pcolormesh(time, depth2, w_currentT, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(-0.15, 0.10)  # Why different from others?
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=14)
    # plt.title("Vitesse verticale corrigée w(z,t)", size=14)
    # plt.xlabel("Temps [h]", size=14)
    # plt.ylabel("Profondeur [m]", size=14)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    # # plt.gca().xaxis.set_visible(False)  # Hides the x-axis
    # # savefig('T'+ str(p) + '_2' + '.png', bbox_inches='tight')
    # # plt.close()




# # ==========================================================================
# # E.    Histograms
# # ==========================================================================

# # ==============================================
# # E1. Histogram of corrected vertical velocities
# # ==============================================

    w_average2 = []
    for j in range(yL):
        Σwmat = wmat_current[j]
        Σwmat = Σwmat[~pd.isnull(Σwmat)]
        w_average2.append(sum(Σwmat) / len(Σwmat))

    for i in range(xL):
        for j in range(yL):
            w_average2.append(wmat_current[j, i])

    # fig3 = plt.figure(3, figsize=(10, 6))
    # plt.hist(w_average2, bins=200, range=(-0.2, 0.2), color='C0')
    # plt.xlabel("Vitesse verticale w [m/s]", size=15)
    # # plt.title("Histogram of corrected vertical velocities of "
    # #           "the transect n°" + str(p), size=14)
    # # savefig('T'+ str(p) + '_3' + '.png', bbox_inches='tight')
    # # plt.close()



# # ===============================================
# # E2. Histogram for a single point in the anomaly
# # ===============================================

    # for i in range(xL):  #120
    #     w1_current = w[120] - w_average[120]

    # fig4 = plt.figure(4, figsize=(10, 6))
    # plt.hist(w1_current, bins=60, range=(-0.2, 0.2))
    # plt.xlabel("Vitesse verticale w [m/s]", size=15)
    # # plt.title('Histogram of corrected vertical velocities for a single point'
    # #           ' of the transect n°' + str(p), size=14)
    # # savefig('T'+ str(p) + '_4' + '.png', bbox_inches='tight')
    # # plt.close()



# # ===========================================================================
# # F.    Calculation of the average corrected vertical velocity for each depth
# # ===========================================================================

    w_average3 = []
    for j in range(1, yL):
        Σwmat = wmat_current[j]
        Σwmat = Σwmat[~pd.isnull(Σwmat)]
        w_average3.append(sum(Σwmat) / len(Σwmat))

    depth1 = depth[1:]

    # # Plot of w average corrected function of depth
    # fig5 = plt.figure(5, figsize=(6, 12))
    # plt.plot(w_average3, depth1)
    # plt.gca().invert_yaxis()

    # plt.ylabel("Profondeur [m]", size=15)
    # plt.xlabel("Vitesse verticale w [m/s]", size=15)
    # plt.title("Corrected current average vertical velocity as a function "
    #           "of depth for the transect n°" + str(p), size=14)
    # plt.gca().xaxis.tick_top()
    # plt.gca().xaxis.set_label_position('top')
    # # savefig('T'+ str(p) + '_5' + '.png', bbox_inches='tight')
    # # plt.close()



# # ==========================================================================
# # G.    Moving average
# # ==========================================================================

    move = 20
    w_average_mov = []
    wmat_currentT = np.transpose(wmat_current)   # (xL, yL) and not (yL, xL)

    for j in range(yL):
        xx = wmat_currentT[:, j]
        mov_avg = np.array([np.nanmean(xx[idx:idx + move-1]) for idx in range
                            (len(xx) - move)])
    # # warning for transects 1, 7, 9 and 13 ok bc mean of nan near the coasts
    # warnings.filterwarnings(action = 'ignore', message = 'Mean of empty slice')

        w_average_mov.append(mov_avg)

    timemov = mytime0[0:-move]
    timem, depthm = np.meshgrid(timemov, depth)
    
#     fig6 = plt.figure(6, figsize=(12, 6))
#     gs2 = fig6.add_gridspec(2, 1)
#     fig6.subplots_adjust(hspace=0.5)
#     fig6.set_size_inches(8, 6)
# #     plt.suptitle("Moving average of the current vertical velocity "
# #                   "for the transect n°" + str(p), size=16)


#     fig6.add_subplot(gs2[0, 0])
#     plt.pcolormesh(time, depth2, w_currentT, shading='auto')
#     plt.gca().invert_yaxis()
#     plt.clim(-0.15, 0.10)
#     cbar = plt.colorbar(aspect=10)
#     cbar.set_label(label="Vitesse [m/s]", size=15)
# #     plt.title("Vitesse verticale du courant pour le transect n°" + str(p), size=15)
#     plt.xlabel("Temps [h]", size=15)
#     plt.ylabel("Profondeur [m]", size=15)
#     plt.gca().xaxis.set_major_formatter(xfmt)


#     fig6.add_subplot(gs2[1, 0])
#     plt.pcolormesh(timem, depthm, w_average_mov, shading='nearest')
#     plt.gca().invert_yaxis()
#     plt.clim(-0.15, 0.10)
#     cbar = plt.colorbar(aspect=10)
#     cbar.set_label(label="Vitesse [m/s]", size=15)
# #     plt.title("Moving average of the current vertical velocity w_current",
# #               size=15)
#     plt.xlabel("Temps [h]", size=15)
#     plt.ylabel("Profondeur [m]", size=15)
#     plt.gca().xaxis.set_major_formatter(xfmt)
#     # savefig('T'+ str(p) + '_6' + '.png', bbox_inches='tight')
#     # plt.close()






# # ==========================================================================
# # H.    VELOCITY ANOMALIES
# # ==========================================================================

# # ===================================================
# # H1. We are trying to establish a threshold to see
# #     where and when the blue anomalies begin and end
# # ===================================================

    THRESHOLD = -0.05  # Threshold at -0.05 at first sight
    wmat1_current = wmat_current[wmat_current < THRESHOLD]

    # # Histogram of the values < -0.05
    # fig7 = plt.figure(7, figsize=(10, 6))
    # plt.hist(wmat1_current, bins=60, range=(-0.2, 0.2))
    # plt.xlabel('Vitesse verticale w [m/s]', size=15)
    # plt.title('Histograme des vitesses verticales corrigées <'
    #           ' -5 cm/s pour le transect n°' + str(p), size=15)
    # # savefig('T'+ str(p) + '_7' + '.png', bbox_inches='tight')
    # # plt.close()



# # ===========================================================================
# # H2. Plot

# # We want to plot the values inferior to -0.05 by replacing the others by nan
# # np.where replaces the values of wmat_current > -0.05 by "np.nan".
# # We want a threshold between -0.15 and -0.05 to have the "little bump" of
# # the histograms. These values represent the anomaly of decreasing velocities
# # ===========================================================================

    wmat_ano = np.where(wmat_current < THRESHOLD, wmat_current, np.nan)

    # fig8 = plt.figure(8, figsize=(12, 6))
    # plt.pcolormesh(time, depth2, wmat_ano, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(-0.15, 0.10)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=15)
    # plt.title("Vitesses verticales < -5 cm/s", size=15)
    # plt.xlabel("Temps [h]", size=15)
    # plt.ylabel("Profondeur [m]", size=15)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    # # savefig('T'+ str(p) + '_8' + '.png', bbox_inches='tight')
    # # plt.close()



# # ========================================================================
# # H3. Test to clean the plot

# # We try to clean the plot by removing the isolated "points" at the bottom
# # and on the left. For that, we put nan at i when i+1, i+2, ..., i+5 = nan
# # ========================================================================

#     wmat2_current = wmat_current
#     for i in range (0, xL-1):
#         for j in range (0, yL-7):
#             if (pd.isnull(wmat_current[j+1, i])
#                 & (pd.isnull(wmat_current[j+2, i]))
#                 & (pd.isnull(wmat_current[j+3, i]))
#                 & (pd.isnull(wmat_current[j+4, i]))
#                 & (pd.isnull(wmat_current[j+5, i]))):
# 
#                 wmat2_current[j:, i] = np.nan
# # It does not "clean" enough.



# # ========================================================================
# # H3.2 Second test to clean the plot

# # We change the threshold to -0.03 m/s: With a colorbar up to -0.03 we can
# # better see the variations in the anomaly: blue "spot" in the center.
# # ========================================================================

    THRESHOLD = -0.03
    CMIN = -0.15
    wmat2_ano = np.where(wmat_current < THRESHOLD, wmat_current, np.nan)

    # fig9 = plt.figure(9, figsize=(12, 6))
    # plt.pcolormesh(time, depth2, wmat2_ano, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(CMIN, THRESHOLD)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=15)
    # plt.title("Vitesses verticales < -3 cm/s", size=15)
    # plt.xlabel("Temps [h]", size=15)
    # plt.ylabel("Profondeur [m]", size=15)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    # # savefig('T'+ str(p) + '_9' + '.png', bbox_inches='tight')
    # # plt.close()



# # =========================================================================
# # H4. Same threshold with the moving average: For the transect 1, we can no
# # longer see the descent, but we can see a real block without any hole
# # =========================================================================

    wmat_average_mov = np.asarray(w_average_mov)  # Convert into a matrix
    wmat_average_mov_ano = np.where(wmat_average_mov < THRESHOLD,
                                    wmat_average_mov, np.nan)

    # fig10 = plt.figure(10, figsize=(12, 6))
    # plt.pcolormesh(timem, depthm, wmat_average_mov_ano, shading='auto')
    # plt.gca().invert_yaxis()
    # plt.clim(CMIN, THRESHOLD)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=15)
    # plt.title("Moyenne mobile des vitesses verticales < -3 cm/s", size=15)
    # plt.xlabel("Temps  [h]", size=15)
    # plt.ylabel("Profondeur [m]", size=15)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    # # savefig('T'+ str(p) + '_10' + '.png', bbox_inches='tight')
    # # plt.close()



# # =======================================================
# # H5. Size grid (for a better selection of the anomalies)
# # =======================================================

# We take rectangles of size length*width
    LENGTH = 20   # 20 at first
    WIDTH = 6     # 6 at first

# Matrix filled with 0 which we replace progressively by
# rectangles of 1's if they contain more than 70% of values
    CRITERIA = 3/10    # 3/10 at first (30% nan, 70% values)
    grid = np.zeros((yL, xL))

    # slide on yL (n bins = from 0 to yL-(n-1))
    for j in range(yL - (WIDTH-1)):
        # slide on xL (n bins = from 0 to xL-(n-1))
        for i in range(xL - (LENGTH-1)):

            # empty rectangle
            rectangle = []

            # we fill it with the values of wmat2_ano on LENGTH*WIDTH defined
            rectangle = wmat2_ano[j:j+WIDTH, i:i+LENGTH]

            # transform the rectangle into a table with panda
            df = pd.DataFrame(rectangle)

            # transforms to True/False with np.nan = True
            df2 = df.isin([np.nan])

            # counts "True", if "True" i.e. nan < 3/10th: matrix = 1
            if np.count_nonzero(df2) < ((WIDTH * LENGTH) * CRITERIA):
                grid[j:j+WIDTH, i:i+LENGTH] = 1

    # Matrix filled with nan
    wmat3_ano = np.full((yL, xL), np.nan)
    # for example: np.full([height, width, 9], np.nan)
    
    # When grid = 1, we put the values of wmat2_ano, otherwise we put nan
    wmat3_ano = np.where(grid == 1, wmat2_ano, np.nan)

  # Plot wmat3_ano i.e. wmat2_ano (wmat_current < -0.03) cleaned by the grid:
    # fig11 = plt.figure(11, figsize=(12, 6))
    # plt.pcolormesh(time, depth2, wmat3_ano, shading='auto')
    # plt.gca().invert_yaxis()
    
    # plt.clim(CMIN, THRESHOLD)
    # cbar = plt.colorbar(aspect=10)
    # cbar.set_label(label="Vitesse [m/s]", size=15)

    # plt.xlabel("Temps [h]", size=15)
    # plt.gca().xaxis.set_tick_params(labelsize=15)
    # plt.gca().xaxis.set_major_formatter(xfmt)
    # plt.ylabel("Profondeur [m]", size=15)
    # plt.gca().yaxis.set_tick_params(labelsize=15)
    # plt.title("Anomalie de vitesses < -3 cm/s pour le transect n°" +str(p), size=14)
    # # savefig('T'+ str(p) + '_11' + '.png', bbox_inches='tight')
    # # plt.close()



# # ==============
# # H6. Histograms
# # ==============

    wmat3_ano_fla = wmat3_ano.flatten()  # flatten(?) the matrix
    wmat_current_fla = wmat_current.flatten()

    # fig12 = plt.figure(12, figsize=(10, 6))

    # # Histogram of the vertical velocities on the whole transect (blue)
    # plt.hist(wmat_current_fla, bins=100, range=(-0.25, 0.25), color='C0')  # color='blue'
    # plt.xlabel('Vitesse verticale w [m/s]', size=20)
    # plt.gca().xaxis.set_tick_params(labelsize=15)
    # plt.gca().yaxis.set_tick_params(labelsize=15)
    # plt.title(" Histogram of the vertical velocities on the whole transect"
    #           " for the transect n°" + str(p), size=14)

    MEAN_HIST12=np.nanmean(wmat_current_fla)
    STD_HIST12=np.nanstd(wmat_current_fla)
    # print('Without anomaly: mean=',MEAN_HIST12,"std=", STD_HIST12)


    # # Histogram of vertical velocity anomaly (red)
    # plt.hist(wmat3_ano_fla, bins=100, range=(-0.25, 0.25), color='red')
    # plt.xlabel('Vitesse verticale w [m/s]', size=20)
    # plt.gca().xaxis.set_tick_params(labelsize=15)
    # plt.gca().yaxis.set_tick_params(labelsize=15)
    # plt.title("Histogram of vertical velocities (blue) and vertical velocities"
    #           " of anomalies (red) for the transect n°" + str(p), size=14)
    # # savefig('T'+ str(p) + '_12' + '.png', bbox_inches='tight')
    # # plt.close()


    # Histogram of velocities with the anomaly removed
    wmat5_ano = np.where(grid == 1, np.nan, wmat_current)
    wmat5_ano_fla = wmat5_ano.flatten()

    # fig13 = plt.figure(13, figsize=(10, 6))
    # plt.hist(wmat5_ano_fla, bins=100, range=(-0.25, 0.25), color='C0')
    # plt.xlabel('Vitesse verticale w [m/s]', size=20)
    # plt.gca().xaxis.set_tick_params(labelsize=15)
    # plt.gca().yaxis.set_tick_params(labelsize=15)
    # plt.title("Histogram of velocities without the anomaly"
    #           " for the transect n°" + str(p), size=14)
    # # savefig('T'+ str(p) + '_13' + '.png', bbox_inches='tight')
    # # plt.close()


# Anomaly vertical velocities:
# < 0 or 2 or whatever because nan are not values so we are sure it will take
# only the values
    w_anomalie = np.where(wmat3_ano < 2)



# # ==============================================
# # H7. Map of all transects with anomalies in red
# # ==============================================

    fig14 = plt.figure(14, figsize=(13, 7))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([5.5, 10, 42.5, 44.5])  # 5, 10, 42.5, 44.5 normal, square 7.5, 9.5, 43, 44.5
    ax.coastlines(resolution='auto', color='k')

    B = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                      ylocs=[42.66666667, 43, 43.33333333, 43.66666667, 44,
                            44.33333333], dms=True, linewidth=0.5,
                      color='grey', alpha=1, linestyle='--')
    # dms minutes, seconds: 20' and 40', (xlocs: the values we want to put)
    # xlocs=[6, 7, 8, 9, 10],

    B.top_labels = False
    B.right_labels = False
    B.xlabel_style = {'size': 15}  # 'color': 'k'}
    B.ylabel_style = {'size': 15}

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                facecolor='lightgrey'))
    ax.add_feature(cfeature.OCEAN, facecolor='skyblue')

    # Trace the transect in blue or dodgerblue
    plt.scatter(lon, lat, color="blue", s=2, alpha=0.5,
                transform=ccrs.PlateCarree())     

    # Trace the anomaly in red
    # plt.scatter(lon[w_anomalie[1]], lat[w_anomalie[1]], color="red", s=2,
    #             alpha=0.5, transform=ccrs.PlateCarree())

    # plt.title("Map of the transect with the anomaly in red", size=14)
    plt.figtext(0.475, 0.04, 'Longitude', size=20)
    plt.figtext(0.1, 0.45, 'Latitude', rotation=90, size=20)

    # Black point for the start of the transect
    plt.scatter(lon[0], lat[0], color='k',  s=20, alpha=1,
                transform=ccrs.PlateCarree())

    # Black arrow for the start of the transect
    # plt.quiver(lon[0], lat[0], lon[20]-lon[0], lat[20]-lat[0], color='k',
    #             width=0.003, scale=10, transform=ccrs.PlateCarree())
    # scale: a smaller scale parameter makes the arrow longer (scale=8 then 6)

    plt.text(lon[0], lat[0]-0.08, '' + str(p), size=12, transform=ccrs.PlateCarree())
                            # -0.06 square  # -0.08 normal



# # ====================================
# # H8. Timeline of transects (in black)
# # ====================================

    frise = mdates.DateFormatter('%d/%m %H:%M')

    # fig15 = plt.figure(15, figsize=(12, 2))
    # plt.title("Timeline of the transect and the anomaly (in red)", size=14)
    # plt.plot([mytime1, mytime2], [0, 0], marker='|',  color='dodgerblue',
    #           markersize=10, linewidth=3)
    # plt.gca().xaxis.set_tick_params(labelsize=15)
    # plt.gca().xaxis.set_major_formatter(frise)
    # # plt.gca().xaxis.set_major_formatter('minuit')
    # plt.grid()





# # if an anomaly exist, do:
    
    w_anomalie_array=np.array(w_anomalie)
    if (w_anomalie_array.size > 0):

        MINIMUM = min(w_anomalie[1])  # 1st value = minimum
        mytime[MINIMUM]  # Time of the minimum
        start = datenum_to_datetime(mytime[MINIMUM])  # Converted into YYYY-MM-DD

        MAXIMUM = max(w_anomalie[1])  # Last value = maximum
        mytime[MAXIMUM]
        end = datenum_to_datetime(mytime[MAXIMUM])

        duration = end - start
        # print("Time of the anomaly:", duration, "start:", strat, "end:", end)

        average_depth = (sum(w_anomalie[0])/len(w_anomalie[0])) * 8 + 19  # 1st bin at 19m, every 8m
        min_depth = min(w_anomalie[0]) * 8 + 19
        max_depth = max(w_anomalie[0]) * 8 + 19
        # print("Average depth:", average_depth, "m, min:", min_depth, "m, max:", max_depth, "m")


        indices_max = np.argwhere(w_anomalie_array[1] == np.amax(w_anomalie_array[1]))
        end_depth = w_anomalie_array[0][indices_max[-1]] * 8 + 19

        indices_min = np.argwhere(w_anomalie_array[1] == np.amin(w_anomalie_array[1]))
        start_depth = w_anomalie_array[0][indices_min[0]] * 8 + 19
        # print("Start depth:", start_depth, "m and end depth:", end_depth, "m")



# H8. Timeline of start and end of the anomalies (in red)
        # plt.figure(15)
        # plt.plot([start, end], [0, 0], color='red', linewidth=3)  # marker='|' markersize=20
        # # savefig('T'+ str(p) + '_15' + '.png', bbox_inches='tight')


# # H7. Start of the anomaly in black on the map
        # plt.figure(14)
        # plt.scatter(lon[MINIMUM], lat[MINIMUM],s=100, marker="*", color='maroon',
        #             transform=ccrs.PlateCarree())
        # # savefig('carte des anomalies, mer en bleue' + '.png', bbox_inches='tight')
        # print('Anomaly beginning : lat, lon:' + str(p), lat[MINIMUM], lon[MINIMUM])
        # print('End:' + str(p), lat[MAXIMUM], lon[MAXIMUM])


# # H9. Ship velocity
#         plt.plot(w_average)
#         plt.scatter(MINIMUM, w_average[MINIMUM], marker="*", color='r')
#         plt.scatter(MAXIMUM, w_average[MAXIMUM], marker="*", color='r')


# # H10. Velocity min, max, mean
        w_min = np.nanmin(wmat3_ano)
        w_max = np.nanmax(wmat3_ano)
        w_mean = np.nanmean(wmat3_ano)
        w_std = np.nanstd(wmat3_ano)
        # print("Mean of the anomaly velocity:", w_mean, ", std:", w_std,"min:", w_min,
        #       "max:", w_max)



# # ===============================================
# # H11. Calculation of mean and standard deviation
# # ===============================================

# replace p with 1, 2, 3, ..., 13 to sum up all the values in one list

# # ANOMALY
        ano_t1=wmat3_ano_fla
        mean_ano_t1=np.nanmean(ano_t1)
        std_ano_t1=np.nanstd(ano_t1)

# # for all the campaign
        # ano_tt=np.hstack((ano_t1, ano_t3, ano_t6))
        # mean_ano_tt=np.nanmean(ano_tt)
        # std_ano_tt=np.nanstd(ano_tt)


# # ALL THE TRANSECT
    all_t1=(w_average2)

    # all_tt=np.hstack((all_t1, all_t2, ...
    # mean_all_tt=np.nanmean(all_tt)
    # std_all_tt=np.nanstd(all_tt)


# # WITHOUT ANOMALY
    without_t1=(wmat_current_fla)
    # without_tt=np.hstack((without_t1, without_t2, ...
    # mean_without_tt=np.nanmean(without_tt)
    # std_without_tt=np.nanstd(without_tt)
