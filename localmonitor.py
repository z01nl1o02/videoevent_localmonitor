import os,sys,pdb,cv2,math,pickle
import numpy as np

"""
papers:

Robust Real-Time Unusual Event Detection Using Multiple Fixed-Location Monitors"
by Amit Adam, Ehud Rivlin, llan Shimshoni, David Reinitz

dataset:
UCSD_Anomaly Dataset


keys:
1. windows size should be selected acoording to application. 8 for ssd radius and neighbour radius is a good value to start testing
2. only observation obeys following two conditions will be inserted to history or check anomaly
   a) most likely and ambiguity test
   b) nonzero optical flow (minima ssd is just center of neighbor window)
   any observation out of these conditions will be ignored 
"""

class LOCAL_MONITOR:
    def __init__(self, position, videoshape, monitor_radius, templ_radius, b_speed_mode = 1, alarm_thresh = 0.8):
    #position: (x,y), monitor center
    #videoshape: shape of video (h,w)
    #monitor_radius: monitor radius
    #templ_radius : template radius in SSD
    #b_speed_mode : speed mode or orientation mode
        self.x, self.y = position
        height, width = videoshape
        self.left = np.maximum(self.x - monitor_radius, monitor_radius)
        self.top = np.maximum(self.y - monitor_radius, monitor_radius)
        self.right = np.minimum(self.x + monitor_radius, width - monitor_radius)
        self.bottom = np.minimum(self.y + monitor_radius, height - monitor_radius)
        self.templ_radius = templ_radius
        self.monitor_radius = monitor_radius
        self.b_speed_mode = b_speed_mode
        self.alarm_thresh = alarm_thresh

        #parameter to normalize SSD value
        self.ssd_a = 1 / 1.0
        self.ssd_k = 1.0

        #threshold to judge ambiguity of observation
        if b_speed_mode == 1:
            self.hist_bin_step = 1.0
            self.hist_bin_num = np.int32(self.monitor_radius / self.hist_bin_step + 0.5)
            self.ambiguity_thresh = 1.5 / self.hist_bin_step
        else:
            self.hist_bin_step = 30.0
            self.hist_bin_num = np.int32(360.0 / self.hist_bin_step + 0.5)
            self.ambiguity_thresh = 20.0 / self.hist_bin_step

        #size of observation buffer
        self.histcapacity = 10 * 60
        self.hists = []

      
    def get_position(self):
        return (self.x, self.y)
 
    def get_region(self):
        return (self.left, self.top, self.right, self.bottom) 

    def is_fitted(self):
        return len(self.hists) >= self.histcapacity  
      
    def calc_ssd(self, f0, f1):
        winsize = (2 * self.templ_radius + 1) * (2 * self.templ_radius + 1) * 1.0
        probmap = np.zeros((self.bottom - self.top, self.right - self.left))

        x = np.int32((self.left + self.right) / 2)
        y = np.int32((self.top + self.bottom) / 2)
        x0 = x - self.templ_radius
        x1 = x + self.templ_radius
        y0 = y - self.templ_radius
        y1 = y + self.templ_radius
        b0 = np.float32(f0[y0:y1,x0:x1])
        
        for y in range(self.top, self.bottom):
            for x in range(self.left, self.right):
                x0 = x - self.templ_radius
                x1 = x + self.templ_radius
                y0 = y - self.templ_radius
                y1 = y + self.templ_radius
                b1 = np.float32(f1[y0:y1,x0:x1])
                ssd = np.mean(np.abs(b0 - b1))
                probmap[y-self.top,x-self.left] = ssd

        probmap = self.ssd_k * np.exp(-self.ssd_a * probmap)  
        if 0:
            cv2.imwrite('f0.jpg', f0)
            cv2.imwrite('f1.jpg', f1)
            img = np.zeros(f0.shape)
            for y in range(probmap.shape[0]):
                for x in range(probmap.shape[1]):
                    row = y + self.top
                    col = x + self.left
                    img[row,col] = np.uint8(probmap[y,x] * 255)
            cv2.imwrite('ssd.jpg', img)

        return probmap

    def histogram_on_speed(self, probmap):
        cx = probmap.shape[1] / 2
        cy = probmap.shape[0] / 2

        binsize = self.hist_bin_step
        binnum = self.hist_bin_num
        hist = np.zeros((binnum,1))

        if probmap[cy,cx] == probmap.max():
            return hist # no optical flow found and this observation will be discarded

        for y in range(probmap.shape[0]):
            for x in range(probmap.shape[1]):
                if probmap[y,x] < 0.0001:
                    continue 
                dx = np.abs(x - cx)
                dy = np.abs(y - cy)
                d = np.maximum( dx, dy ) #block distance
                d = np.int64(d/binsize)
                if d >= binnum:
                    d = binnum - 1
                hist[d,0] += probmap[y,x]
        hist = hist /(0.001 + np.sum(hist) )
        return hist
        
    def histogram_on_orientation(self, probmap):
        cx = probmap.shape[1] / 2
        cy = probmap.shape[0] / 2
        binsize = self.hist_bin_step
        binnum = self.hist_bin_num
        hist = np.zeros((binnum,1))
        if probmap[cy,cx] == probmap.max():
            return hist # no optical flow found and this observation will be discarded
        for y in range(probmap.shape[0]):
            for x in range(probmap.shape[1]):
                if probmap[y,x] < 0.0001:
                    continue 
                dx = x - cx
                dy = y - cy
                a = math.atan2(dy,dx) * 180 / np.pi
                if  a < 0:
                    a += 360
                a = np.int32(a / binsize)
                if a >= binnum:
                    a = binnum - 1
                hist[a,0] += probmap[y,x]
        hist = hist / (np.sum(hist) + 0.0001)
        return hist

    def calc_histogram(self,probmap):
        if self.b_speed_mode == 1:
            return self.histogram_on_speed(probmap)
        else:
            return self.histogram_on_orientation(probmap)

    #a method to show monitor inforamtion stored
    def calc_histogram_mean(self):
        if len(self.hists) < 1:
            return 0.0
        refhist = np.zeros(self.hists[0].shape)
        for h in self.hists:
            refhist += h
        refhist /= len(self.hists)      
        
        s = 0
        for k in range(refhist.shape[0]):
            s += (k + 1) * refhist[k,0]
        s = s * 1.0 / refhist.shape[0]
        return s

    def most_likely_and_ambiguity_test(self, hist):
        m1 = hist.max()
        if m1 < 0.0001:
            return (0,0) #no optical flow found 

        [y,x] = np.nonzero(hist == m1)
        y = y[0]
        x = x[0]
        if type(y) is np.ndarray:
            y = y[0]
            x = x[0]
        yml = y

        d = [k - yml for k in range(hist.shape[0])]
        d = np.array(d)
        d = np.abs(d)
        d = np.reshape(d, (hist.shape[0], 1))
        if np.sum(d * hist) >= self.ambiguity_thresh:
            return (0,yml) #bad observation
        return (1,yml)

 
    def calc_anomaly_probability(self, queryhist):
        #if len(self.hists) < self.histcapacity and self.histcapacity > 0:
        #     return -1.0

        if len(self.hists) < 5:
            return -1.0 #too many bad observation so you can't set too high threshold here

        #most-likely and ambiguity test
        ret,yml = self.most_likely_and_ambiguity_test(queryhist)
        if ret == 0:
            return -1.0

        #get reference histogram
        refhist = np.zeros(queryhist.shape)
        for h in self.hists:
            refhist += h
        refhist /= len(self.hists)         
        return 1 - refhist[yml,0]

    #online learn     
    def fit_predict(self, f0, f1):
        probmap = self.calc_ssd(f0,f1)
        hist = self.calc_histogram(probmap)
        prob = self.calc_anomaly_probability(hist)
        if prob >= 0:
            if len(self.hists) >= self.histcapacity and self.histcapacity > 0:
                self.hists.pop() #delete the last one      
            self.hists.insert(0, hist) #insert the header

        if prob > self.alarm_thresh:
            return 1 #alarmed
        elif prob < 0:
            return -1 #bad observation
        else:
            return 0

    #offline learn 
    def fit(self, f0, f1):
        probmap = self.calc_ssd(f0,f1)
        hist = self.calc_histogram(probmap)

        #only insert good observation
        ret, yml = self.most_likely_and_ambiguity_test(hist)
        if ret == 0:
            return 

        if len(self.hists) >= self.histcapacity and self.histcapacity > 0:
            self.hists.pop() #delete the last one      
        self.hists.insert(0, hist) #insert the header

    #online predict
    def predict(self, f0, f1):
        if len(self.hists) < 1:
            return -1

        probmap = self.calc_ssd(f0,f1)
        hist = self.calc_histogram(probmap)
        prob = self.calc_anomaly_probability(hist)

        if prob > self.alarm_thresh:
            return 1 #alarmed
        elif prob < 0:
            return -1
        else:
            return 0


def scan_dir_for(dirname,objext): 
    results = []
    for rdir,pdir, names in os.walk(dirname):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 == cmp(ext, objext):
                fname = os.path.join(rdir,name)
                results.append((sname, fname))
    return results


def setup_monitors(img):
    results = []
    monitor_radius = 8
    templ_radius = 8
    frameshape = img.shape
    b_speed_mode  = 1
    alarm_thresh = 0.8
    for y in range(monitor_radius + templ_radius, img.shape[0] - monitor_radius - templ_radius, 2 * monitor_radius):
        for x in range(monitor_radius + templ_radius, img.shape[1] - monitor_radius - templ_radius, 2 * monitor_radius):
            position = (x,y)
            monitor = LOCAL_MONITOR(position, frameshape, monitor_radius, templ_radius,b_speed_mode, alarm_thresh)
            results.append(monitor)
    return results 

def run_train(traindir, monitors):
    filenames = scan_dir_for(traindir, '.tif')
    for idx in range(len(filenames)):
        sname, fname = filenames[idx]
        f1 = cv2.imread(fname,0) 
        if len(monitors) < 1:
            monitors = setup_monitors(f1)
            print 'setup ', len(monitors)
        if idx == 0:
            f0 = f1
            continue
        for k in range(len(monitors)):
            monitors[k].fit(f0,f1)
        f0 = f1 #switch 
        print '.',
    print '\r\n'
    return monitors

def run_online_train(imgdir,outdir):
    filenames = scan_dir_for(imgdir, '.tif')
    monitors = [] 
    for idx in range(len(filenames)):
        sname, fname = filenames[idx]
        f1 = cv2.imread(fname, 0)
        if len(monitors) < 1:
            monitors = setup_monitors(f1)
            print 'setup ', len(monitors)
        if idx == 0:
            f0 = f1
            continue

        fitted = [0 for k in range(len(monitors))]
        alarms = [0 for k in range(len(monitors))]
        for k in range(len(monitors)):
            alarms[k] = monitors[k].fit_predict(f0,f1)
            fitted = monitors[k].is_fitted()
        
        if 0:
            img = np.zeros(f1.shape)
            for mts in monitors:
                v = mts.calc_histogram_peak()
                left,top,right,bottom = mts.get_region()
                img[top:bottom, left:right] = v
            cv2.imwrite('out/%s.1.jpg'%sname, img)

        f0 = f1 #switch 
  
        if 1:
            show_model(monitors) 

        if 1:
            with open('online_model%d.txt'%k, 'w') as f:
                pickle.dump(monitors, f) #save model

      
        if np.sum(fitted) * 10 < len(fitted) * 3:
            print 'fited: ',np.sum(fitted),'/',len(fitted)
            continue

        img = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
        maskcolor = np.array([0,0,255])
        for k in range(len(alarms)):
            if alarms[k] <= 0:
                continue
            cx,cy = monitors[k].get_centerxy()
            radius = 8
            w0 = 0.4
            w1 = 1 - w0
            for y in range(cy - radius, cy + radius,1):
                for x in range(cx - radius, cx + radius,1):
                    img[y,x,:] = np.uint8(img[y,x,:] * w0 + maskcolor * w1)
        outfilename = outdir + sname + ".jpg"
        cv2.imwrite(outfilename, img)
        print 'online train ',sname
    return monitors


def run_predict(traindir, outdir, monitors):

    filenames = scan_dir_for(traindir, '.tif')
    
    for idx in range(len(filenames)):
        sname, fname = filenames[idx]
        f1 = cv2.imread(fname, 0)
        if idx == 0:
            f0 = f1
            continue

        if 0:
            img = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
            for k in range(len(monitors)):
                x,y = monitors[k].get_position()
                cv2.putText(img, '%d'%k ,(x,y), cv2.FONT_HERSHEY_COMPLEX,0.2,(255,0,0))
            cv2.imwrite('test.2.jpg', img)

        alarmed = [0 for k in range(len(monitors))]
        for k in range(len(monitors)):
            alarmed[k] = monitors[k].predict(f0,f1)
   
        alarmed = np.array(alarmed) 
        total = len(alarmed)
        quiet = np.sum(alarmed < 0)
         
        f0 = f1 #switch 
        img = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) 
        maskcolor = np.array([0,0,255])
        for k in range(len(alarmed)):
            if alarmed[k] <= 0:
                continue
            cx,cy = monitors[k].get_position()
            radius = 8
            w0 = 0.4
            w1 = 1 - w0
            for y in range(cy - radius, cy + radius,1):
                for x in range(cx - radius, cx + radius,1):
                    img[y,x,:] = np.uint8(img[y,x,:] * w0 + maskcolor * w1)
        outfilename = outdir + sname + ".jpg"
        print 'predict',sname,' %d/%d'%(quiet,total)
        cv2.imwrite(outfilename, img)
    return monitors


def show_model(monitors):
    monitor_infos = []
    m1 = 0
    for mts in monitors:
        m = mts.calc_histogram_mean()
        left,top,right,bottom = mts.get_region()
        monitor_infos.append((m, left, top, right, bottom))
        if m > m1:
            m1 = m
    img = np.zeros((158,238))
    for item in monitor_infos:
        m, left, top, right, bottom = item
        img[top:bottom, left:right] = np.uint8(m * 255.0 / m1)
    cv2.imwrite('test.1.jpg', img)



if __name__ == "__main__":
    with open('config.txt', 'r') as f:
        rootdir = f.readline().strip()
    if len(sys.argv) >= 2:
        if 0 == cmp(sys.argv[1], '-train'):
            monitors = run_train(rootdir+'Train/Train001/', [])
            for k in range(2, 34):
                traindir = rootdir + 'Train/Train%.3d/'%k
                monitors = run_train(traindir,monitors)
                with open('model%d.txt'%k, 'w') as f:
                    pickle.dump(monitors, f) #save model
                print k, '/34'
                if 0: #debug
                    show_model(monitors)
        elif 0 == cmp(sys.argv[1], '-predict'):
            modelpath = sys.argv[2]
            testdir = rootdir+sys.argv[3]
            outdir = sys.argv[4]
            with open(modelpath, 'r') as f:
                monitors = pickle.load(f)
            if 1:
                show_model(monitors)
            run_predict(testdir,outdir, monitors)
        elif 0 == cmp(sys.argv[1], '-olearn'):
            testdir = sys.argv[2]
            outdir = sys.argv[3]
            monitors = run_online_train(testdir, outdir)
        else:
            print 'train/predict/olearn'

    
                 



