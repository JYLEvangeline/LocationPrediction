import string
import datetime
import numpy as np
import pickle
import math
import os
import re
hour_gap = 6
valid_portion = 0.1
test_portion = 0.2
user_session_min = 3 #!!! previous 3
session_len_min = 2
session_len_4sq_max = 20
session_len_gb_max = 40



def prepare_data(root_path, small_path=None, u_cnt_max = -1, task=0):
    print 'task: ', task
    if small_path is not None and not os.path.exists(small_path):
        os.mkdir(small_path)
    if task == -1:
        read_checkins('tweets-cikm.txt', 'venues.txt', 'loc-foursquare_totalCheckins.txt')
        #read_checkins('/Users/quanyuan/Dropbox/Research/Spatial/checkins.csv', '/Users/quanyuan/Dropbox/Research/Spatial/venues.csv', '/Users/quanyuan/Dropbox/Research/Spatial/loc-foursquare_totalCheckins.txt')
    if task == 0:
        for dataset in ['foursquare', 'gowalla', 'brightkite']:
            print 'dataset: ', dataset
            # find records of frequent users in NYC ----delete the one lower than 10
            process_raw_file(root_path + 'loc-' + dataset + '_totalCheckins.txt', root_path + dataset + '_total.txt',
                                u_freq_min=10, v_freq_min=10 if dataset == 'foursquare' else 5, lat_min=40.4774, lat_max=40.9176, lng_min=-74.2589,
                                lng_max=-73.7004)   #!!!jiayi u_freq_min = 10 ,v_freq_min = 10
            # set to different session, I need to adjust the value later
            generate_session_file(root_path + dataset + '_total.txt', root_path + dataset + '_session.txt')
    if task == 1:
        for dataset in ['foursquare', 'gowalla', 'brightkite']:
            save_path = small_path + dataset + '/'
            if save_path is not None and not os.path.exists(save_path):
                os.mkdir(save_path)
            print 'dataset: ', dataset
            dl = DataLoader(hour_gap=6, offset=300)
            dl.add_records(root_path + dataset + '_session.txt', save_path + 'dl.pk', u_cnt_max)
            dl.summarize()
            f = open(root_path + 'blacklist_' + dataset + '.txt', 'w', -1)
            for uid, records_u in dl.uid_records.items():
                if not records_u.valid():
                    f.write(dl.uid_u[uid] + '\n')
            f.close()
    if task == 2:
        for dataset in ['foursquare', 'gowalla', 'brightkite']:
            save_path = small_path + dataset + '/'
            if save_path is not None and not os.path.exists(save_path):
                os.mkdir(save_path)
            print 'dataset: ', dataset
            blacklist = set()
            f = open(root_path + 'blacklist_' + dataset + '.txt', 'r', -1)
            for l in f:
                blacklist.add(l.strip())
            f.close()
            dl = DataLoader(hour_gap=6, offset=300)
            dl.add_records(root_path + dataset + '_session.txt', save_path + 'dl.pk', u_cnt_max=u_cnt_max,
                           blacklist=blacklist)
            dl.summarize()
            dl.show_info()
        # for uid, records_u in dl.uid_records.items():
        #     for rid, record in enumerate(dl.uid_records[uid].records):
        #         record.peek()
        #         if rid < dl.uid_records[uid].test_idx:
        #             print "train"
        #         else:
        #             print "test"

class DataLoader(object):
    def __init__(self, hour_gap=6, offset=0):
        self.hour_gap = hour_gap
        self.u_uid = {}
        self.uid_u = {}
        self.v_vid = {}
        self.uid_records = {}
        self.nu = 0
        self.nv = 0
        self.nt = 24 * 2
        self.nr = 0
        self.vid_coor = {}
        self.vid_coor_rad = {}
        self.vid_coor_nor = {}
        self.vid_coor_nor_rectified = {}
        self.vid_pop = {}
        self.sampling_list = []
        self.offset = offset

    def summarize(self):
        for uid, record_u in self.uid_records.items():
            record_u.summarize()

    def add_records(self, file_path, dl_save_path, u_cnt_max=-1, blacklist=None):
        f = open(file_path, 'r', -1)
        for line in f:
            al = line.strip().split('\t')
            u = al[0]
            if blacklist is not None and u in blacklist:
                continue
            v = al[4]
            dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%dT%H:%M:%SZ') - datetime.timedelta(minutes=self.offset)
            start_2009 = datetime.datetime.strptime('2009-03-08T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2009 = datetime.datetime.strptime('2009-11-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            start_2010 = datetime.datetime.strptime('2010-03-14T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2010 = datetime.datetime.strptime('2010-11-07T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            start_2011 = datetime.datetime.strptime('2011-03-13T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2011 = datetime.datetime.strptime('2011-11-06T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            start_2012 = datetime.datetime.strptime('2012-03-11T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2012 = datetime.datetime.strptime('2012-11-04T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            year = dt.year
            if year == 2009 and dt > start_2009 and dt < end_2009 or \
                year == 2010 and dt > start_2010 and dt < end_2010 or \
                year == 2011 and dt > start_2011 and dt < end_2011 or \
                year == 2012 and dt > start_2012 and dt < end_2012:
                dt = dt + datetime.timedelta(minutes=60)

            lat = string.atof(al[2])
            lng = string.atof(al[3])
            if u not in self.u_uid:
                if u_cnt_max > 0 and len(self.u_uid) >= u_cnt_max:
                    break
                # print u, self.nu
                self.u_uid[u] = self.nu
                self.uid_u[self.nu] = u
                self.uid_records[self.nu] = UserRecords(self.nu)
                self.nu += 1
            if v not in self.v_vid:
                self.v_vid[v] = self.nv
                self.vid_pop[self.nv] = 0
                self.vid_coor_rad[self.nv] = np.array([np.radians(lat), np.radians(lng)])
                self.vid_coor[self.nv] = np.array([lat, lng])
                self.nv += 1
            uid = self.u_uid[u]
            vid = self.v_vid[v]
            self.sampling_list.append(vid)
            self.vid_pop[vid] += 1
            self.uid_records[uid].add_record(dt, uid, vid, self.nr)
            self.nr += 1
        f.close()

        coor_mean = np.zeros(2)
        coor_var = np.zeros(2)
        for vid, coor in self.vid_coor.items():
            coor_mean += coor
        coor_mean /= len(self.vid_coor)
        for vid, coor in self.vid_coor.items():
            coor_var += (coor - coor_mean) ** 2
        coor_var /= len(self.vid_coor)
        coor_var = np.sqrt(coor_var)
        for vid in self.vid_coor:
            self.vid_coor_nor[vid] = (self.vid_coor[vid] - coor_mean) / coor_var
            lat_sub = self.vid_coor[vid][0] - coor_mean[0]
            lng_sub = self.vid_coor[vid][1] - coor_mean[1]
            lat_rectified = lat_sub / coor_var[0]
            lng_rectified = lng_sub * math.cos(self.vid_coor[vid][0]) / coor_var[0]
            self.vid_coor_nor_rectified[vid] = np.array([lat_rectified, lng_rectified])
        if blacklist is not None:
            pickle.dump(self, open(dl_save_path, 'wb'))

    def show_info(self):
        print 'U: ', self.nu, 'V: ', self.nv, 'R: ', self.nr, 'T: ', self.nt

    def write_to_files(self, root_path):
        self.summarize()
        # f_coor_nor = open(root_path + "coor_nor.txt", 'w')
        f_train = open(root_path + "train.txt", 'w')
        f_test = open(root_path + "test.txt", 'w')
        for uid, records_u in self.uid_records.items():
            vids_long = [[], []]
            vids_short_al = [[], []]
            tids = [[], []]
            vids_next = [[], []]
            tids_next = [[], []]
            for rid, record in enumerate(records_u.records):
                if record.is_first:
                    vids_short = []
                vids_short.append(record.vid)
                if rid < records_u.test_idx:
                    vids_long[0].append(record.vid)
                    tids[0].append(record.tid)
                    vids_next[0].append(record.vid_next)
                    tids_next[0].append(record.tid_next)
                vids_long[1].append(record.vid)
                tids[1].append(record.tid)
                vids_next[1].append(record.vid_next)
                tids_next[1].append(record.tid_next)
                if record.is_last:
                    if rid < records_u.test_idx:
                        vids_short_al[0].append(vids_short)
                    vids_short_al[1].append(vids_short)
                    vids_short = []
                #
                #
                #
                #
                # role_id = 0 if rid < records_u.test_idx else 1
                # if record.is_first:
                #     vids_short = []
                # vids_long[role_id].append(record.vid)
                # vids_short.append(record.vid)
                # tids[role_id].append(record.tid)
                # vids_next[role_id].append(record.vid_next)
                # tids_next[role_id].append(record.tid_next)
                # if record.is_last:
                #     vids_short_al[role_id].append(vids_short)
                #     vids_short = []
            f_train.write(str(uid) + ',' + str(len(vids_short_al[0])) + ',' + str(records_u.test_idx) + '\n')
            f_test.write(str(uid) + ',' + str(len(vids_short_al[1])) + ',' + str(records_u.test_idx) + '\n')
            f_train.write(','.join([str(vid) for vid in vids_long[0]]) + '\n')
            f_test.write(','.join([str(vid) for vid in vids_long[1]]) + '\n')
            for vids_short in vids_short_al[0]:
                f_train.write(','.join([str(vid) for vid in vids_short]) + '\n')
            for vids_short in vids_short_al[1]:
                f_test.write(','.join([str(vid) for vid in vids_short]) + '\n')
            f_train.write(','.join([str(tid) for tid in tids[0]]) + '\n')
            f_test.write(','.join([str(tid) for tid in tids[1]]) + '\n')
            f_train.write(','.join([str(vid) for vid in vids_next[0]]) + '\n')
            f_test.write(','.join([str(vid) for vid in vids_next[1]]) + '\n')
            f_train.write(','.join([str(tid) for tid in tids_next[0]]) + '\n')
            f_test.write(','.join([str(tid) for tid in tids_next[1]]) + '\n')
        coor_nor = np.zeros((len(self.vid_coor_nor), 2), dtype=np.float64)
        for vid in range(self.nv):
            coor_nor[vid] = self.vid_coor_nor[vid]
        np.savetxt(root_path + 'coor_nor.txt', coor_nor, fmt="%lf", delimiter=',')
        # coor_nor.tofile(root_path + 'coor_nor.txt', sep=',')
            # f_coor_nor.write(','.join([str(coor) for coor in self.vid_coor_nor[vid]]) + '\n')
        f_train.close()
        f_test.close()
        # f_coor_nor.close()
        f_u = open(root_path + "u.txt", 'w')
        f_v = open(root_path + "v.txt", 'w')
        f_t = open(root_path + "t.txt", 'w')
        for u in self.u_uid:
            f_u.write(u + ',' + str(self.u_uid[u]) + '\n')
        for v in self.v_vid:
            f_v.write(v + ',' + str(self.v_vid[v]) + '\n')
        for t in xrange(48):
            f_t.write(str(t) + ',' + str(t) + '\n')
        f_u.close()
        f_v.close()
        f_t.close()

class Record(object):
    def __init__(self, dt, uid, vid, vid_next=-1, tid_next = -1, is_first=False, is_last=False, rid=None):
        self.dt = dt
        self.rid = rid
        self.uid = uid
        self.vid = vid
        self.tid = dt.hour
        if dt.weekday > 4 or dt.weekday == 4 and dt.hour>=18:
            self.tid += 24
        self.tid_168 = dt.weekday() * 24 + dt.hour
        self.vid_next = vid_next
        self.tid_next = tid_next
        self.is_first = is_first
        self.is_last = is_last

    def peek(self):
        print 'u: ', self.uid, '\tv: ', self.vid, '\tt: ', self.tid, '\tvid_next: ', self.vid_next, '\tis_first: ', self.is_first, '\tis_last: ', self.is_last, 'dt: ', self.dt, 'rid: ', self.rid

class UserRecords(object):
    def __init__(self, uid):
        self.uid = uid
        self.records = []
        self.dt_last = None
        self.test_idx = 0

    def add_record(self, dt, uid, vid, rid=None):
        is_first = False
        if self.dt_last is None or (dt - self.dt_last).total_seconds() / 3600.0 > hour_gap:
            is_first = True
            if len(self.records) > 0:
                self.records[len(self.records) - 1].is_last = True
        record = Record(dt, uid, vid, is_first=is_first, is_last=True, rid=rid)
        if len(self.records) > 0:
            self.records[len(self.records) - 1].vid_next = record.vid
            self.records[len(self.records) - 1].tid_next = record.tid
            if not is_first:
                self.records[len(self.records) - 1].is_last = False
            else:
                self.records[len(self.records) - 1].vid_next = -1
        self.records.append(record)
        self.dt_last = dt
        self.is_valid = True

    def summarize(self):
        session_begin_idxs = []
        session_len = 0
        session_begin_idx = 0
        for rid, record in enumerate(self.records):
            if record.is_first:
                session_begin_idx = rid
            session_len += 1
            if record.is_last:
                if session_len >= 2:
                    session_begin_idxs.append(session_begin_idx)
                session_len = 0
        if len(session_begin_idxs) < 2:
            self.is_valid = False
            return
        test_session_idx = int(len(session_begin_idxs) * (1 - test_portion))
        if test_session_idx == 0:
            test_session_idx = 1
        if test_session_idx < len(session_begin_idxs):
            self.test_idx = session_begin_idxs[test_session_idx]
        else:
            self.is_valid = False


    def valid(self):
        return self.is_valid

    def get_records(self, mod=0):
        if mod == 0:  # train only
            return self.records[0: self.test_idx]
        elif mod == 1:  # test only
            return self.records[self.test_idx: len(self.records)]
        else:
            return self.records

    def get_predicting_records_cnt(self, mod=0):
        cnt = 0
        if mod == 0:  # train only
            for record in self.records[0: self.test_idx]:
                if record.is_last:
                    continue
                cnt += 1
            return cnt
        else:  # test only
            for record in self.records[self.test_idx: len(self.records)]:
                if record.is_last:
                    continue
                cnt += 1
            return cnt





def generate_session_file(file_path, save_path, hour_gap=6):
    session_len_max = session_len_4sq_max if file_path.find('foursquare') >= 0 else session_len_gb_max
    fr = open(file_path, 'rt', -1)
    fw = open(save_path, 'wt')
    u_session = []
    session = []
    u_pre = 'start'
    dt_pre = None
    u_set = set()
    for line in fr:
        al = line.split('\t')
        u = al[0]
        u_set.add(u)
        dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%dT%H:%M:%SZ')
        if u != u_pre:
            session_valid_cnt = 0
            for session in u_session:
                if len(session) >= session_len_min:
                    session_valid_cnt += 1
            if session_valid_cnt >= user_session_min:
                for session in u_session:
                    if len(session) >= session_len_min and len(session) <= session_len_max:
                        for l in session:
                            fw.write(l + '\n')
            u_session = []
            session = []
            dt_pre = None
        if dt_pre is None:
            dt_pre = dt
        if (dt - dt_pre).total_seconds() / 3600.0 > hour_gap:
            # if len(session) >= 2:
            u_session.append(session)
            session = []
        session.append(line.strip())
        u_pre = u
        dt_pre = dt
    u_session.append(session)
    session_valid_cnt = 0
    for session in u_session:
        if len(session) >= session_len_min:
            session_valid_cnt += 1
    if session_valid_cnt >= user_session_min:
        for session in u_session:
            if len(session) >= session_len_min:
                for l in session:
                    fw.write(l + '\n')
    fr.close()
    fw.close()

def process_raw_file(file_path, save_path, u_freq_min=10, v_freq_min = 10, lat_min=-180, lat_max=180, lng_min=-180, lng_max=180):
    u_freq = {}
    v_freq = {}
    f = open(file_path, 'r', -1)
    for line in f:
        al = line.strip().split('\t')
        u = al[0]
        u_freq[u] = 1 if u not in u_freq else u_freq[u] + 1
    f.close()
    f = open(file_path, 'r', -1)
    for line in f:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        if u_freq[u] < u_freq_min:
            continue
        v_freq[v] = 1 if v not in v_freq else v_freq[v] + 1
    f.close()
    u_set = set()
    fr = open(file_path, 'rt')
    fw = open(save_path, 'wt')
    lines = []
    v_set = set()
    u_pre = '0'
    for line in fr:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        lat = string.atof(al[2])
        lng = string.atof(al[3])
        if lat < lat_min or lat > lat_max or lng < lng_min or lng > lng_max: #only take consider this square. why? !!!
            continue
        if u_freq[u] < u_freq_min or v_freq[v] < v_freq_min:
            continue
        v_set.add(v)
        if u == u_pre:
            lines.append(line)
        else:
            u_set.add(u)
            for l in lines[::-1] if file_path.find('foursquare') == -1 else lines:
                fw.write(l)
            lines = [line]
            u_pre = u
    u_set.add(u)
    for l in lines[::-1] if file_path.find('foursquare') == -1 else lines:
        fw.write(l)
    fr.close()
    fw.close()
    print 'U: ', len(u_set)
    print 'V: ', len(v_set)


def read_checkins(checkins_path, venue_path, save_path):
    # read venue coordinates
    f = open(venue_path, 'r')
    venue_lat = {}
    venue_lng = {}
    for line in f:
        al = line.strip().split(",")
        venue = al[0]
        lat = string.atof(al[1])
        lng = string.atof(al[2])
        venue_lat[venue] = lat
        venue_lng[venue] = lng
    f.close()
    # read checkins
    f = open(checkins_path, 'r')
    u_checkins = {}
    cnt = 0
    # lines = f.readlines()
    for line in f:
        checkin = CheckIn(line.strip(), venue_lat, venue_lng)
        if checkin.isvalid():
            if checkin.user not in u_checkins:
                u_checkins[checkin.user] = []
            u_checkins[checkin.user].append(checkin)
            cnt += 1
            if cnt % 1000 == 0:
                print 'reading: %d' % cnt
    f.close()
    # threads = []
    # threads_num = 8
    # per_thread = len(lines) / threads_num
    # print per_thread
    # for i in range(threads_num):
    #     if threads_num - i == 1:
    #         t = threading.Thread(target=process, args=(i, lines[i * per_thread:], venue_lat, venue_lng, u_checkins))
    #     else:
    #         t = threading.Thread(target=process, args=(i, lines[i * per_thread:i * per_thread + per_thread], venue_lat, venue_lng, u_checkins))
    #     threads.append(t)
    # for i in range(threads_num):
    #     threads[i].start()
    # for i in range(threads_num):
    #     threads[i].join()
    checkins_all = []
    for checkins_u in u_checkins.values():
        checkins_u = sorted(checkins_u, cmp=lambda x, y: cmp(x.dt, y.dt))
        for checkin in checkins_u:
            checkins_all.append(checkin)
    f = open(save_path, 'w')
    for checkin in checkins_all:
        f.write(checkin.to_string() + '\n')
    f.close()

class CheckIn(object):
    def __init__(self, str, venue_lat, venue_lng):
        try:
            # print str
            self.is_valid = True
            al = str.replace('\n', '').split('')
            #self.user = al[7].strip('"') #quan
            self.user = al[1].strip('"') #jiayi
            text_temp = re.match(r'(.*)http(.*?)//*', al[7].strip('"'), re.M | re.I) #jiayi
            self.text = text_temp.group(1) #jiayi
            self.venue = al[8].strip('"')
            if self.venue not in venue_lng.keys():
                self.is_valid = False
            self.lat = venue_lat[self.venue]
            self.lng = venue_lng[self.venue]
            self.dt = datetime.datetime.strptime(al[4].strip('"'),'%Y-%m-%d %H:%M:%S') #jiayi
            #self.dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%d %H:%M:%S')#quan
            self.time = al[4].strip('"').replace(' ', 'T') + 'Z'  #jiayi
            #self.time = al[1].strip('"').replace(' ', 'T') + 'Z' #quan
        except:
            self.is_valid = False

    def isvalid(self):
        return self.is_valid

    def to_string(self):
        #return self.user + '\t' + self.time + '\t' + str(self.lat) + '\t' + str(self.lng) + '\t' + self.venue
        return self.user + '\t' + self.time + '\t' + str(self.lat) + '\t' + str(self.lng) + '\t' + self.venue + '\t' +self.text


def analyze_time_dist(root_path):
    for dataset in ['foursquare', 'gowalla', 'brightkite']:
        print dataset
        #dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb')) #!!!
        dl = pickle.load(open(root_path + dataset + '/' + 'dl.pk', 'rb'))
        tid_cnt = {}
        for _, records_u in dl.uid_records.items():
            for record in records_u.records:
                tid = record.tid % 24
                if tid not in tid_cnt:
                    tid_cnt[tid] = 0
                tid_cnt[tid] += 1
        for tid in sorted(tid_cnt.keys()):
            print tid, tid_cnt[tid]

def analyze_session_len(root_path):
    for dataset in ['foursquare', 'gowalla', 'brightkite']:
        print dataset
        #dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb')) #!!!
        dl = pickle.load(open(root_path + dataset + '/' + 'dl.pk', 'rb'))
        len_cnt = {}
        for _, records_u in dl.uid_records.items():
            for record in records_u.records:
                if record.is_first:
                    len = 0
                len += 1
                if record.is_last:
                    if len not in len_cnt:
                        len_cnt[len] = 0
                    len_cnt[len] += 1
        for len in len_cnt:
            print len, len_cnt[len]

def transform_data(root_path):
    for dataset in ['foursquare', 'gowalla', 'brightkite']:
        save_path = root_path + dataset + '/'
        dl = pickle.load(open(save_path + 'dl.pk', 'rb'))
        dl.write_to_files(save_path)
        # if dataset == 'foursquare':
        #     for record in dl.uid_records[0].records:
        #         record.peek()

def distance_data(root_path):
    coor_file = root_path + 'coor_nor.txt'
    f_distance = open(root_path + "distance.txt", 'w')
    vid_coor_nor = np.loadtxt(coor_file, delimiter=',', dtype=np.float64)
    for i in range(len(vid_coor_nor)-1):
        distance = []
        for j in range(i + 1, len(vid_coor_nor)):
            distance.append(np.sqrt(np.sum((vid_coor_nor[i] - vid_coor_nor[j]) ** 2)))
        f_distance.write(','.join([str(d) for d in distance]) + '\n')

        print(i)
    f_distance.close()
    return vid_coor_nor

def read_distance(root_path):
    result = []
    i=0
    with open(root_path+"distance.txt", 'r') as f:
        for line in f.readlines():
            print(i)
            i = i+1
            al = line.strip().split(',')
            al_float = map(eval,al)
            result.append(al_float)
    return result

if __name__ == "__main__":
    root_path = '/Users/quanyuan/Dropbox/Research/LocationCuda/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationCuda/') \
        else 'data/'
    #prepare_data(root_path, task=-1) #1
    #prepare_data(root_path, root_path + 'full/', task=0) #2
    #prepare_data(root_path, root_path + 'full/', task=1)  #3
    #prepare_data(root_path, root_path + 'full/', task=2)  #4
    #analyze_time_dist(root_path + 'full/') #5?
    #analyze_session_len(root_path + 'full/') #6?
    #transform_data(root_path + 'full/') #7?
    coor=distance_data('LocationCuda/' + 'small/foursquare/')
    coor = distance_data('LocationCuda/' + 'full/foursquare/')
    #result = read_distance('LocationCuda/' + 'small/foursquare/')

    #distance_data('LocationCuda/' + 'full/foursquare/')


    # prepare_data(root_path, root_path + 'full/', task=2)
    #prepare_data(root_path, root_path + 'small/', u_cnt_max=500, task=2)
    # analyze_time_dist(root_path + 'full/')
    # analyze_session_len(root_path + 'full/')
    #transform_data(root_path + 'small/')
    #
