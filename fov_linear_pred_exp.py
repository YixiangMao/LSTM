# experiment on several linear FoV prodiction algorithm. 
# @Liyang's Truncated Linear: truncate at sign of Delta_fov changes.
# @Chenge's Interpolated Univariate Spline: Only use most recent 2 point to spline and extrapolate
# Yao's Truncated Linear: truncate when Delta_fov changes largely. Speed_up method: test 30, 15, 8 ... points to truncate

import numpy as np
import matplotlib.pylab as plt
import pickle
import math
# import padasip as pa
# import utilities as uti
from scipy.interpolate import InterpolatedUnivariateSpline


def load_viewport(vp_trace, FRAME_RATE):
	yaw_trace = []
	pitch_trace = []
	file = open(vp_trace, 'rb')
	data = pickle.load(file)
	file.close()
	video_length = len(data)
	scd = round(video_length/FRAME_RATE) + 1
	#print(len(data[0][1]))
	for i in range(scd):  
		for j in range(1,FRAME_RATE):
			yaw_trace.append((data[i][j][1][0]/math.pi)*180.0 + 180.0)
			pitch_trace.append((data[i][j][1][1]/math.pi)*180.0 + 90.0)
	return yaw_trace, pitch_trace


def process_fov_trace_test(yaw_trace, pitch_trace, mult):
	
	#smooth the FoV trace using Filman Filter
	time_trace = []
	for i in range (len(yaw_trace)):
	    time_trace.append(i)
	kf = kalman_filter(time_trace, yaw_trace, pitch_trace)
	kf.init_kf()
	time_gap = 1
	kf_x = kf.kf_run(time_gap, kf_predict=False)
	yaw_trace = [math.floor(p[0]%360.0) for p in kf_x]
	pitch_trace = [math.floor(p[1]) for p in kf_x]

	#extend the trace by flipping and repeat the smoothed trace
	VIDEO_LEN = len(yaw_trace) * mult
	yaw_trace = yaw_trace + yaw_trace[::-1]
	pitch_trace = pitch_trace + pitch_trace[::-1]
	fov_len = len(yaw_trace)
	new_yaw = []
	new_pitch = []
	number = VIDEO_LEN // fov_len
	for i in range (number+1):
		new_yaw = new_yaw + yaw_trace
		new_pitch = new_pitch + pitch_trace
	return new_yaw, new_pitch

def predict_yaw_trun_new_fast(yaw_trace, ind, history_length, prediction_duration):
    # yaw_trace: list, all integer numbers
    # predict prediction_duration frames from ind-th frame.
    # history_length is the maximum history duration the predicter use, may not useful.
	if ind ==0:
		yaw_trace[0] = INIT_YAW
		return yaw_trace,0
	elif ind-history_length<0:
		offset = 0
		history_length = ind
		pred_yaw_data = [yaw_trace[offset]]
		pred_yaw_idx = [offset]
		for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
			if yaw_trace[offset+i+1] - pred_yaw_data[i] >= 180:
				pred_yaw_data.append(yaw_trace[offset+i+1]-360)
			elif yaw_trace[offset+i+1] - pred_yaw_data[i] <= -180:
				pred_yaw_data.append(yaw_trace[offset+i+1]+360)
			else:
				pred_yaw_data.append(yaw_trace[offset+i+1])
			pred_yaw_idx.append(offset+i+1)
	else:
	    offset = ind-history_length
	    pred_yaw_data = [yaw_trace[offset]]
	    pred_yaw_idx = [offset]
	    for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
	        if yaw_trace[offset+i+1] - pred_yaw_data[i] >= 180:
	            pred_yaw_data.append(yaw_trace[offset+i+1]-360)
	        elif yaw_trace[offset+i+1] - pred_yaw_data[i] <= -180:
	            pred_yaw_data.append(yaw_trace[offset+i+1]+360)
	        else:
	            pred_yaw_data.append(yaw_trace[offset+i+1])
	        pred_yaw_idx.append(offset+i+1)

    # Truncate 
	threshold = 10.0

	new_value = [pred_yaw_data[-(ii+1)] for ii in range(len(pred_yaw_data))]
	new_index = [pred_yaw_idx[-(ii+1)] for ii in range(len(pred_yaw_idx))]
	print(ind, " check point:", new_value, " indx: ", new_index)

	# print(pred_yaw_idx)
	# print(new_index)
	an = len(new_index)
	current_predict_model = np.polyfit(new_index, new_value, 1)
	status = True
	for ii in range(an):
	    if	np.abs(new_value[ii] - np.polyval(current_predict_model,new_index[ii]))>threshold:
	        print("outlier0: ", ii, new_value[ii] , np.polyval(current_predict_model,new_index[ii]))
	        status = False
	while status == False and an>2:
	    an = int(np.ceil(an/2))
	    new_value = [new_value[ii] for ii in range(an)]
	    new_index = [new_index[ii] for ii in range(an)]
	    current_predict_model = np.polyfit(new_index, new_value, 1)
	    status = True
	    for ii in range(an):
	        if	np.abs(new_value[ii] - np.polyval(current_predict_model,new_index[ii]))>threshold:
	            print("outlier: ", ii, new_value[ii] , np.polyval(current_predict_model,new_index[ii]))
	            status = False
	# new_value = [pred_yaw_data[-1], pred_yaw_data[-2]]
	# new_index = [pred_yaw_idx[-1], pred_yaw_idx[-2]]
	# an = len(new_index)
	# current_predict_model = np.polyfit(new_index, new_value, 1)	
	# for i in reversed(range(len(pred_yaw_data)-2)):
	# 	predict_value = np.polyval(current_predict_model,pred_yaw_idx[i])%360.0
	# 	if np.abs(pred_yaw_data[i] - predict_value) < threshold :
	# 	    new_value.append(pred_yaw_data[i])
	# 	    new_index.append(pred_yaw_idx[i])
	# 	    current_predict_model = np.polyfit(new_index, new_value, 1)	
	# 	    # temp = pred_yaw_data[i]
	# 	else:
	# 	    break
	# an = len(new_index)
	new_value.reverse()
	new_index.reverse()
	# print(new_index[-1])
	assert new_index[-1] == ind
	# print(new_index)
	weight = [np.exp(-0.5*(an-1-ind)) for ind in range(an)]

	yaw_predict_model = np.polyfit(new_index, new_value, 1, w = weight)	
	for i in range(prediction_duration):
	    yaw_trace[ind+1+i] = np.polyval(yaw_predict_model,ind+1+i)%360.0
	return yaw_trace,an # return the same yaw trace list, directly replace the values from ind-th to (ind + prediction_duration -1)-th frame 	


def predict_yaw_trun_new(yaw_trace, ind, history_length, prediction_duration):
    # yaw_trace: list, all integer numbers
    # predict prediction_duration frames from ind-th frame.
    # history_length is the maximum history duration the predicter use, may not useful.
	if ind ==0:
		yaw_trace[0] = INIT_YAW
		return yaw_trace,0
	elif ind-history_length<0:
		offset = 0
		history_length = ind
		pred_yaw_data = [yaw_trace[offset]]
		pred_yaw_idx = [offset]
		for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
			if yaw_trace[offset+i+1] - pred_yaw_data[i] >= 180:
				pred_yaw_data.append(yaw_trace[offset+i+1]-360)
			elif yaw_trace[offset+i+1] - pred_yaw_data[i] <= -180:
				pred_yaw_data.append(yaw_trace[offset+i+1]+360)
			else:
				pred_yaw_data.append(yaw_trace[offset+i+1])
			pred_yaw_idx.append(offset+i+1)
	else:
	    offset = ind-history_length
	    pred_yaw_data = [yaw_trace[offset]]
	    pred_yaw_idx = [offset]
	    for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
	        if yaw_trace[offset+i+1] - pred_yaw_data[i] >= 180:
	            pred_yaw_data.append(yaw_trace[offset+i+1]-360)
	        elif yaw_trace[offset+i+1] - pred_yaw_data[i] <= -180:
	            pred_yaw_data.append(yaw_trace[offset+i+1]+360)
	        else:
	            pred_yaw_data.append(yaw_trace[offset+i+1])
	        pred_yaw_idx.append(offset+i+1)

    # Truncate 
	new_value = [pred_yaw_data[-1], pred_yaw_data[-2]]
	new_index = [pred_yaw_idx[-1], pred_yaw_idx[-2]]
	# sign = np.sign(pred_yaw_data[-1] - pred_yaw_data[-2])
	# difference = pred_yaw_data[-1] - pred_yaw_data[-2]
	# threshold = np.abs(0.3 * difference)
    # temp = pred_yaw_data[-2]
	an = len(new_index)
	threshold = 10.0
	current_predict_model = np.polyfit(new_index, new_value, 1)	
	for i in reversed(range(len(pred_yaw_data)-2)):
		predict_value = np.polyval(current_predict_model,pred_yaw_idx[i])%360.0
		if np.abs(pred_yaw_data[i] - predict_value) < threshold :
		# if np.sign(temp - pred_yaw_data[i]) == sign or np.abs(temp - pred_yaw_data[i])<0.01:    
		# if np.abs((temp-pred_yaw_data[i]) - difference) < threshold:# or (threshold<6.0 and np.abs(temp - pred_yaw_data[i])<10.0):
		    new_value.append(pred_yaw_data[i])
		    new_index.append(pred_yaw_idx[i])
		    current_predict_model = np.polyfit(new_index, new_value, 1)	
		    # temp = pred_yaw_data[i]
		else:
		    break
	an = len(new_index)
	new_value.reverse()
	new_index.reverse()
	assert new_index[-1] == ind
	# print(new_index)
	weight = [np.exp(-0.5*(an-1-ind)) for ind in range(an)]

	yaw_predict_model = np.polyfit(new_index, new_value, 1, w = weight)	
	for i in range(prediction_duration):
	    yaw_trace[ind+1+i] = np.polyval(yaw_predict_model,ind+1+i)%360.0
	return yaw_trace,an # return the same yaw trace list, directly replace the values from ind-th to (ind + prediction_duration -1)-th frame 	




def predict_yaw_trun(yaw_trace, ind, history_length, prediction_duration):
    # yaw_trace: list, all integer numbers
    # predict prediction_duration frames from ind-th frame.
    # history_length is the maximum history duration the predicter use, may not useful.
	if ind ==0:
		yaw_trace[0] = INIT_YAW
		return yaw_trace,0
	elif ind-history_length<0:
		offset = 0
		history_length = ind
		pred_yaw_data = [yaw_trace[offset]]
		pred_yaw_idx = [offset]
		for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
			if yaw_trace[offset+i+1] - pred_yaw_data[i] >= 180:
				pred_yaw_data.append(yaw_trace[offset+i+1]-360)
			elif yaw_trace[offset+i+1] - pred_yaw_data[i] <= -180:
				pred_yaw_data.append(yaw_trace[offset+i+1]+360)
			else:
				pred_yaw_data.append(yaw_trace[offset+i+1])
			pred_yaw_idx.append(offset+i+1)
	else:
	    offset = ind-history_length
	    pred_yaw_data = [yaw_trace[offset]]
	    pred_yaw_idx = [offset]
	    for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
	        if yaw_trace[offset+i+1] - pred_yaw_data[i] >= 180:
	            pred_yaw_data.append(yaw_trace[offset+i+1]-360)
	        elif yaw_trace[offset+i+1] - pred_yaw_data[i] <= -180:
	            pred_yaw_data.append(yaw_trace[offset+i+1]+360)
	        else:
	            pred_yaw_data.append(yaw_trace[offset+i+1])
	        pred_yaw_idx.append(offset+i+1)

    # Truncate 
	# print(ind, " check point:", pred_yaw_data)
	new_value = [pred_yaw_data[-1], pred_yaw_data[-2]]
	new_index = [pred_yaw_idx[-1], pred_yaw_idx[-2]]
	an = len(new_index)
	sign = np.sign(pred_yaw_data[-1] - pred_yaw_data[-2])
	# difference = pred_yaw_data[-1] - pred_yaw_data[-2]
	# threshold = np.abs(0.3 * difference)
	temp = pred_yaw_data[-2]
	for i in reversed(range(len(pred_yaw_data)-2)):
		if np.sign(temp - pred_yaw_data[i]) == sign or np.abs(temp - pred_yaw_data[i])<0.01:    
		# if np.abs((temp-pred_yaw_data[i]) - difference) < threshold:# or (threshold<6.0 and np.abs(temp - pred_yaw_data[i])<10.0):
		    new_value.append(pred_yaw_data[i])
		    new_index.append(pred_yaw_idx[i])
		    temp = pred_yaw_data[i]
		else:
		    break
	an = len(new_index)
	new_value.reverse()
	new_index.reverse()
	assert new_index[-1] == ind
	# print(new_value)
	yaw_predict_model = np.polyfit(new_index, new_value, 1)	
	for i in range(prediction_duration):
	    yaw_trace[ind+1+i] = np.polyval(yaw_predict_model,ind+1+i)%360.0
	return yaw_trace, an # return the same yaw trace list, directly replace the values from ind-th to (ind + prediction_duration -1)-th frame 	





def predict_pitch_trun(pitch_trace, ind, history_length, prediction_duration):
    # pitch_trace: list, all integer numbers
    # predict prediction_duration frames from ind-th frame.
    # history_length is the maximum history duration the predicter use, may not useful.
	if ind ==0:
		pitch_trace[0] = INIT_PITCH
		return pitch_trace
	elif ind-history_length<0:
		offset = 0
		history_length = ind
		pred_yaw_data = [pitch_trace[offset]]
		pred_yaw_idx = [offset]
		for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
			pred_yaw_data.append(pitch_trace[offset+i+1])
			pred_yaw_idx.append(offset+i+1)
	else:
	    offset = ind-history_length
	    pred_yaw_data = [pitch_trace[offset]]
	    pred_yaw_idx = [offset]
	    for i in range(history_length):
	        # Assume history_length is always less than or equal to ind
	        pred_yaw_data.append(pitch_trace[offset+i+1])
	        pred_yaw_idx.append(offset+i+1)

    # Truncate 
	new_value = [pred_yaw_data[-1], pred_yaw_data[-2]]
	new_index = [pred_yaw_idx[-1], pred_yaw_idx[-2]]
	sign = np.sign(pred_yaw_data[-1] - pred_yaw_data[-2])
	temp = pred_yaw_data[-2]
	for i in reversed(range(len(pred_yaw_data)-2)):
	    if np.sign(temp - pred_yaw_data[i]) == sign or temp == pred_yaw_data[i]:
	        new_value.append(pred_yaw_data[i])
	        new_index.append(pred_yaw_idx[i])
	        temp = pred_yaw_data[i]
	    else:
	        break	
	new_value.reverse()
	new_index.reverse() 
	assert new_index[-1] == ind
	yaw_predict_model = np.polyfit(new_index, new_value, 1)	
	for i in range(prediction_duration):
		roundvalue = np.polyval(yaw_predict_model,ind+1+i)
		if roundvalue > 180.0:
			roundvalue = 180.0
		if roundvalue < 0.0:
			roundvalue = 0.0
		pitch_trace[ind+1+i] = roundvalue
	return pitch_trace # return the same yaw trace list, directly replace the values from ind-th to (ind + prediction_duration -1)-th frame 	


class kalman_filter(object):
    def __init__(self, time_trace, yaw_trace, pitch_trace):
        self.time_trace = time_trace
        self.yaw_trace = yaw_trace
        self.pitch_trace = pitch_trace
        self.kf_x = []
        self.modified_kf_x = []
        self.Q_v = 0.1
        self.R_v = 0.1

    def init_kf(self):
        self.B = 0
        self.U = 0
        self.P = np.diag((0.01, 0.01, 0.01, 0.01))
        self.dt = self.time_trace[1] - self.time_trace[0]
        assert self.dt > 0 
        self.dx = self.yaw_trace[1] - self.yaw_trace[0]
        self.dy = self.pitch_trace[1] - self.pitch_trace[0]
        self.X = np.array([[self.yaw_trace[1]],
                          [self.pitch_trace[1]],
                          [self.dx/self.dt], 
                          [self.dy/self.dt]])
        # self.X = np.array([[self.yaw_trace[1]],
        #                   [self.pitch_trace[1]],
        #                   [0.0], 
        #                   [0.0]])

        self.A = np.array([[1, 0 , self.dt, 0],
                           [1, 0 , 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Assume measurement does NOT inlucde velocity x/y
        # self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])     # 2*4
        # self.Y = np.array([[0.0], [0.0]])                   # Will udpate later
        # self.Q = np.eye(self.X.shape[0])*self.Q_v           # 4*4
        # self.R = np.eye(self.Y.shape[0])*self.R_v           # 2*2

        # Include vx and vy
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])     # 2*4
        self.Y = np.array([[0.0], [0.0], [0.0], [0.0]])                   # Will udpate later
        self.Q = np.eye(self.X.shape[0])*self.Q_v           # 4*4
        self.R = np.eye(self.Y.shape[0])*self.R_v           # 2*2

    def kf_update(self):
        IM = np.dot(self.H, self.X)                                 # 2*1
        IS = self.R + np.dot(self.H, np.dot(self.P, self.H.T))      # 2*2
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(IS)))     # 4*2
        self.X = self.X + np.dot(K, (self.Y-IM))                    # 4*1
        self.P = self.P - np.dot(K, np.dot(IS, K.T))                # 4*4
        # LH = gauss_pdf(Y, IM, IS) 
        return (K,IM,IS)

    def kf_predict(self):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def kf_update_para(self, i):
        self.dt = self.time_trace[i] - self.time_trace[i-1]
        self.dx = self.yaw_trace[i] - self.yaw_trace[i-1]
        self.dy = self.pitch_trace[i-1] - self.pitch_trace[i-1]
        if not self.dt == 0:
            # Assume there is no vx/vy
            # self.Y = np.array([[self.yaw_trace[i]], 
            #                    [self.pitch_trace[i]]])
            # There are vx/vy
            self.Y = np.array([[self.yaw_trace[i]], 
                               [self.pitch_trace[i]],
                               [self.dx/self.dt],
                               [self.dy/self.dt]])
            self.A = np.array([[1.0, 0 , self.dt, 0],
                                [0, 1.0 , 0, self.dt],
                                [0, 0, 1.0, 0],
                                [0, 0, 0, 1.0]])

    def kf_run(self, time_gap, kf_predict=False):
        for i in range(2, len(self.time_trace)):
            self.kf_update_para(i)
            self.kf_predict()
            # Get predict info before slef.x is updated using measurement Y
            self.kf_x.append(self.X.T[0][:2])
            self.kf_update()
            self.modified_kf_x.append(self.X.T[0][:2])

        if False: #Config.show_kf:
            plt.scatter(self.time_trace[2:], [p for p in self.yaw_trace[2:]], c='r')
            plt.scatter(self.time_trace[2:], [p[0] for p in self.kf_x], c='b')
            plt.scatter(self.time_trace[2:], [p[0] for p in self.modified_kf_x], c='g')
            plt.show()
            input()
            plt.scatter(self.time_trace[2:], [p for p in self.pitch_trace[2:]], c='r')
            plt.scatter(self.time_trace[2:], [p[1] for p in self.kf_x], c='b')
            plt.scatter(self.time_trace[2:], [p[1] for p in self.modified_kf_x], c='g')
            plt.show()
            input()

        if kf_predict:
            new_A = np.array([[1, 0 , time_gap, 0],
                                [0, 1 , 0, time_gap],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            new_Xs = np.dot(new_A, self.X).tolist()
            return new_Xs
        else:
            return self.modified_kf_x




#### main begin here:


DELAY = 3

FRAME_RATE = 30
VIEWPORT_TRACE_FILENAME_NEW = './trace/viewport/0/7.p'
yaw_trace, pitch_trace = load_viewport(VIEWPORT_TRACE_FILENAME_NEW, FRAME_RATE)
yaw_trace = [(yaw-180.0)%360.0 for yaw in yaw_trace]
MULTIPLIER = 2
yaw_trace, pitch_trace = process_fov_trace_test(yaw_trace, pitch_trace, MULTIPLIER) # smooth and extend the FoV trace by times MULTIPLIER
TRACE_LEN = len(yaw_trace)
pred_yaw = [0]*(TRACE_LEN+100)
pred_pitch = [0]*(TRACE_LEN+100)

INIT_YAW = 180.0   # 0 - 360
INIT_PITCH = 90.0  # 0 - 180




####### Interpolated Univariate Spline by Chenge
# pred_yaw_rec = [INIT_YAW]*(TRACE_LEN+100)
# pred_pitch_rec = [INIT_PITCH]*(TRACE_LEN+100)
# for i in range (DELAY+1,TRACE_LEN):
#     if i-30-DELAY<0:
#         pass
#     else:
#         xi = [j-30 for j in range (30)]
#         yi = [yaw_trace[i-30-DELAY+j] for j in range (30)]
#         s = InterpolatedUnivariateSpline(xi, yi, k=1)
#         x = [j for j in range (70)]
#         y = s(x)
#         pred_yaw_rec[i] = y[DELAY-1]
#         yi = [pitch_trace[i-30-DELAY+j] for j in range (30)]
#         s = InterpolatedUnivariateSpline(xi, yi, k=1)
#         y = s(x)
#         pred_pitch_rec[i] = y[DELAY-1]


# timeplot = [i for i in range(TRACE_LEN)]
# plotrangeleft = 0
# plotrangeright = TRACE_LEN
# plt.figure(figsize=(30,12))
# plt.subplot(511);plt.title("FoV traces Chenge");plt.xlim(plotrangeleft,plotrangeright)
# plt.plot(timeplot,yaw_trace,"b", label="Actural yaw");plt.legend()
# plt.plot(timeplot,pred_yaw_rec[:TRACE_LEN],"go-", label="Predicted yaw");plt.legend()
# plt.plot(timeplot,pitch_trace,"r", label="Actural pitch");plt.legend() #;plt.grid(True);plt.show()
# plt.plot(timeplot,pred_pitch_rec[:TRACE_LEN],"m", label="Predicted pitch");plt.legend();plt.grid(True)



# yi = [7,8,9,7.1,6.1,5,4]
# xi = [i-len(yi) for i in range(len(yi))]
# s = InterpolatedUnivariateSpline(xi, yi, k=1)
# x = [i for i in range (5)]
# y = s(x)
# print(y)




####### Truncated Linear predictor Liyang
# pred_yaw_rec = []
# pred_pitch_rec = []
# sample_rec = []
# for i in range(DELAY):
#     pred_yaw_rec.append(INIT_YAW)
#     pred_pitch_rec.append(INIT_PITCH)
#     sample_rec.append(0)
# for i in range(DELAY,TRACE_LEN):
#     # print(i)
#     pred_yaw,sn = predict_yaw_trun(pred_yaw, i-DELAY, 30-DELAY, 70)
#     pred_pitch = predict_pitch_trun(pred_pitch, i-DELAY, 30-DELAY, 70)
#     pred_yaw_rec.append(pred_yaw[i])
#     pred_pitch_rec.append(pred_pitch[i])
#     sample_rec.append(sn)
#     if i+1-DELAY>=0:
#         pred_yaw[i+1-DELAY] = yaw_trace[i+1-DELAY]
#         pred_pitch[i+1-DELAY] = pitch_trace[i+1-DELAY]
# timeplot = [i for i in range(TRACE_LEN)]
# plotrangeleft = 0
# plotrangeright = TRACE_LEN
# # plt.figure(figsize=(30,6))
# plt.subplot(512);plt.title("FoV traces Liyang");plt.xlim(plotrangeleft,plotrangeright)
# plt.plot(timeplot,yaw_trace,"b", label="Actural yaw");plt.legend()
# plt.plot(timeplot,pred_yaw_rec,"go-", label="Predicted yaw");plt.legend()
# plt.plot(timeplot,pitch_trace,"r", label="Actural pitch");plt.legend() #;plt.grid(True);plt.show()
# plt.plot(timeplot,pred_pitch_rec,"m", label="Predicted pitch");plt.legend();plt.grid(True)
# plt.subplot(513);plt.title("prediction sample number");plt.xlim(plotrangeleft,plotrangeright)
# plt.plot(timeplot,sample_rec,"ko-", label="Sample number");plt.legend();plt.grid(True)



#### by Yao
pred_yaw_rec = []
pred_pitch_rec = []
sample_rec = []
for i in range(DELAY):
    pred_yaw_rec.append(INIT_YAW)
    pred_pitch_rec.append(INIT_PITCH)
    sample_rec.append(0)
for i in range(DELAY,TRACE_LEN):
    # print(i)
    pred_yaw,sn = predict_yaw_trun_new(pred_yaw, i-DELAY, 30-DELAY, 70)
    pred_pitch = predict_pitch_trun(pred_pitch, i-DELAY, 30-DELAY, 70)
    pred_yaw_rec.append(pred_yaw[i])
    pred_pitch_rec.append(pred_pitch[i])
    sample_rec.append(sn)
    if i+1-DELAY>=0:
        pred_yaw[i+1-DELAY] = yaw_trace[i+1-DELAY]
        pred_pitch[i+1-DELAY] = pitch_trace[i+1-DELAY]
timeplot = [i for i in range(TRACE_LEN)]
plotrangeleft = 0
plotrangeright = TRACE_LEN
# plt.figure(figsize=(30,6))
plt.figure(figsize=(30,12))
plt.subplot(411);plt.title("Yesterday's algorithm");plt.xlim(plotrangeleft,plotrangeright)
plt.plot(timeplot,yaw_trace,"b", label="Actural yaw");plt.legend()
plt.plot(timeplot,pred_yaw_rec,"go-", label="Predicted yaw");plt.legend()
plt.plot(timeplot,pitch_trace,"r", label="Actural pitch");plt.legend() #;plt.grid(True);plt.show()
plt.plot(timeplot,pred_pitch_rec,"m", label="Predicted pitch");plt.legend();plt.grid(True)
plt.subplot(412);plt.title("prediction sample number");plt.xlim(plotrangeleft,plotrangeright)
plt.plot(timeplot,sample_rec,"ko-", label="Sample number");plt.legend();plt.grid(True)








pred_yaw_rec = []
pred_pitch_rec = []
sample_rec = []
for i in range(DELAY):
    pred_yaw_rec.append(INIT_YAW)
    pred_pitch_rec.append(INIT_PITCH)
    sample_rec.append(0)
for i in range(DELAY,TRACE_LEN):
    # print(i)
    pred_yaw,sn = predict_yaw_trun_new_fast(pred_yaw, i-DELAY, 30-DELAY, 70)
    pred_pitch = predict_pitch_trun(pred_pitch, i-DELAY, 30-DELAY, 70)
    pred_yaw_rec.append(pred_yaw[i])
    pred_pitch_rec.append(pred_pitch[i])
    sample_rec.append(sn)
    if i+1-DELAY>=0:
        pred_yaw[i+1-DELAY] = yaw_trace[i+1-DELAY]
        pred_pitch[i+1-DELAY] = pitch_trace[i+1-DELAY]
timeplot = [i for i in range(TRACE_LEN)]
plotrangeleft = 0
plotrangeright = TRACE_LEN
# plt.figure(figsize=(30,6))
plt.subplot(413);plt.title("Fast method");plt.xlim(plotrangeleft,plotrangeright)
plt.plot(timeplot,yaw_trace,"b", label="Actural yaw");plt.legend()
plt.plot(timeplot,pred_yaw_rec,"go-", label="Predicted yaw");plt.legend()
plt.plot(timeplot,pitch_trace,"r", label="Actural pitch");plt.legend() #;plt.grid(True);plt.show()
plt.plot(timeplot,pred_pitch_rec,"m", label="Predicted pitch");plt.legend();plt.grid(True)
plt.subplot(414);plt.title("prediction sample number");plt.xlim(plotrangeleft,plotrangeright)
plt.plot(timeplot,sample_rec,"ko-", label="Sample number");plt.legend();plt.grid(True);plt.show()






































# x,y,z = uti.reample_traces(3, aa, aa, aa)
# print(x,y,z)


# test = 1
# print(test)
# test += 1
# print(test)
# test += 1
# print(test)
# a = [[1,2,3][4,5,6][7,8,9]]
# b = [[1,2,1][4,5,6][1,2,1]]
# c = 3.1
# print(a-b)
# yaw_trace = [111, 335, 335, 334, 332, 330, 326, 318, 307, 302, 300, 300, 130.02041771471704]
# yaw_trace = uti.predict_yaw_trun(yaw_trace, 10, 10, 1)
# print(yaw_trace)
# print(yaw_trace[10])



# VIEWPORT_TRACE_FILENAME_NEW = './trace/viewport/0/1.p'
# FRAME_RATE = 30
# yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW, FRAME_RATE)
# time_trace = []
# for i in range (len(yaw_trace)):
#     time_trace.append(i)
# kf = uti.kalman_filter(time_trace, yaw_trace, pitch_trace)
# kf.init_kf()
# time_gap = 1
# kf_x = kf.kf_run(time_gap, kf_predict=False)
# # yaw_new = [p[0] for p in kf_x]
# # pitch_new = [p[1] for p in kf_x]
# print (np.floor(kf_x))


# tile_map = []
# line = []
# for j in range (int(8192/256)):  #form a line with all zeros
#     line.append(0)
# for i in range (int(4096/256)): 
#     tile_map.append(line)
# yaw = 180
# pitch = 90

# tile_map, tile_ind, ae, be, ab, bb = uti.update_tile_map_server(tile_map, 0, yaw, pitch, 8, 40, 512)
# print(tile_map)



# data = [1, 0.954839669, 0.930498131, 0.911926245, 0.898636733, 0.888480351, 0.87972517, 0.8688215, 0.862923163, 0.857041807, 0.849908416, 0.844222598, 0.838310905, 0.833840721, 0.828657154, 0.823896984, 0.819463227, 0.815203716, 0.810923265, 0.80685304, 0.803951925, 0.79939608, 0.795769323, 0.792766214, 0.789137248, 0.78480464, 0.781397835, 0.777289468, 0.773129572, 0.769792362, 0.766453108, 0.762785825, 0.759086078, 0.755982748, 0.752956347, 0.749401028, 0.745890136, 0.742094079, 0.73900785, 0.735602908, 0.732430054, 0.729247643, 0.726122988, 0.723290642, 0.720641503, 0.718137907, 0.715342747, 0.712521404, 0.710138356, 0.707532782, 0.705158497, 0.702680663, 0.700496356, 0.698458415, 0.696812428, 0.695588697, 0.69455019, 0.693615012, 0.692147204, 0.690090066]
# dat2 = [1, 0.955714836, 0.922530232, 0.90298182, 0.881384703, 0.865473064, 0.853694196, 0.845787609, 0.838758071, 0.830302269, 0.825012121, 0.818063225, 0.813320271, 0.80830623, 0.802691113, 0.797659049, 0.792898198, 0.790013078, 0.786595883, 0.783224216, 0.78076305, 0.77782983, 0.776710561, 0.774108776, 0.771564394, 0.770209141, 0.767704929, 0.766259356, 0.763745712, 0.761246024, 0.759613686, 0.756671624, 0.755235635, 0.753152168, 0.750977549, 0.74983185, 0.74800321, 0.746753647, 0.74523354, 0.743951467, 0.742746458, 0.740989159, 0.740324912, 0.738382398, 0.736968688, 0.73589238, 0.734716063, 0.733777566, 0.732472318, 0.731698579, 0.730739289, 0.729403732, 0.729062031, 0.727920477, 0.727005095, 0.726581216, 0.725823947, 0.725427108, 0.724916693, 0.724370221]
# print(len(data))

# file_nema = './trace/bandwidth/sprint-downlink.txt' 
# multiple = 1.0
# addition = 0.0
# with open(file_nema) as f:
# 	content = f.readlines()
# 	content = [max(multiple * float(x.strip()) + addition, - (multiple * float(x.strip()) + addition)) for x in content] # scale the trace
# print (content[:250])


# # creation of data
# N = 10
# x = np.random.normal(0, 1, (N, 4)) # input matrix
# v = np.random.normal(0, 0.1, N) # noise
# d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

# # identification
# f = pa.filters.FilterRLS(n=4, mu=0.9, w="random")
# y, e, w = f.run(d, x)

# print (y)
# print (d)
# print (e)
# print (w)

# # show results
# plt.figure(figsize=(15,9))
# plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
# plt.plot(d,"b", label="d - target")
# plt.plot(y,"g", label="y - output");plt.legend()
# plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
# plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
# plt.tight_layout()
# plt.show()


# these two function supplement your online measurment
# def measure_x():
#     # it produces input vector of size 3
#     x = np.random.random(3)
#     return x
    
# def measure_d(x):
#     # meausure system output
#     d = 2*x[0] + 1*x[1] - 1.5*x[2]
#     return d
    
# N = 100
# log_d = np.zeros(N)
# log_y = np.zeros(N)
# filt = pa.filters.FilterRLS(3, mu=0.5)
# for k in range(N):
#     # measure input
#     x = measure_x()
#     print("x.T:", x.T)
#     # predict new value
#     y = filt.predict(x)
#     print("y:", y)
#     # do the important stuff with prediction output
#     pass    
#     # measure output
#     d = measure_d(x)
#     print("d:", d)
#     # update filter
#     filt.adapt(d, x)
#     # log values
#     log_d[k] = d
#     log_y[k] = y
    
# ### show results
# plt.figure(figsize=(15,9))
# plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
# plt.plot(log_d,"b", label="d - target")
# plt.plot(log_y,"g", label="y - output");plt.legend()
# plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
# plt.plot(10*np.log10((log_d-log_y)**2),"r", label="e - error [dB]")
# plt.legend(); plt.tight_layout(); plt.show()
# init_wei = np.array([0.03, 0.07, 0.1, 0.25, 0.55])
# filt = pa.filters.FilterRLS(5, mu=0.99, eps=0.1, w=init_wei)

# x = [24165517.24137931,23561379.31034483,26340413.79310345,27750068.96551724,28797241.37931034]
# x = np.array(x)
# d = 28515310.344827585
# filt.adapt(d, x)
# predidarr = np.array([23561379.31034483,26340413.79310345,27750068.96551724,28797241.37931034,28515310.344827585])
# ave = filt.predict(predidarr)
# print(ave)

# x = [23561379.31034483,26340413.79310345,27750068.96551724,28797241.37931034, 29165517.24137931]
# x = np.array(x)
# d = 29515310.344827585
# filt.adapt(d, x)
# predidarr = np.array([26340413.79310345,27750068.96551724,28797241.37931034, 29165517.24137931,29515310.344827585])
# ave = filt.predict(predidarr)
# print(ave)

# yaw_trace = []
# for i in range (1000):
#     yaw_trace.append(i+1)
# display_time = 3.0
# video_seg_index = 3# video sgement
# VIEW_PRED_SAMPLE_LEN = 10
# FRAME_MV_LIMIT = 100
# POLY_ORDER = 1
# VIDEO_FPS = 30

# yaw_predict_value = 0.0
# if display_time < 1:
# 	yaw_predict_value = 0.0
# else:
#     yaw_trace[80] = -1
#     vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(display_time*VIDEO_FPS)
#     print(vp_index)
#     vp_value = []
#     for index in vp_index:
#     	vp_value.append(yaw_trace[index])
#     print(vp_value)
#     for value in vp_value[1:]:
#         print (value - vp_value[vp_value.index(value)-1])
#         if value - vp_value[vp_value.index(value)-1] > FRAME_MV_LIMIT:
#             value -= 360.0
#         elif vp_value[vp_value.index(value)-1] - value > FRAME_MV_LIMIT:
#             value += 360.0
#     new_value = [vp_value[-1], vp_value[-2]]
#     new_index = [vp_index[-1], vp_index[-2]]
#     sign = np.sign(vp_value[-1] - vp_value[-2])
#     temp = vp_value[-2]
#     sign_index = -3
#     for i in reversed(range(VIEW_PRED_SAMPLE_LEN+sign_index+1)):
#     	if np.sign(temp - vp_value[i]) == sign:
#     		new_value.append(vp_value[i])
#     		new_index.append(vp_index[i])
#     		temp = vp_value[i]
#     	else:
#     		break
#     new_value.reverse()
#     new_index.reverse()	
#     yaw_predict_model = np.polyfit(new_index, new_value, POLY_ORDER)
#     yaw_predict_idx = int(video_seg_index*VIDEO_FPS + VIDEO_FPS/2)
#     yaw_predict_value = np.round(np.polyval(yaw_predict_model,yaw_predict_idx))
# yaw_predict_value %= 360
# print (yaw_predict_value)


