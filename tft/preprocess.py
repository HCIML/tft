import pickle
import pandas as pd
import numpy as np
import datetime
import time
from datetime import datetime, timedelta, date

class PreprocessData():

    def __int__(self):
        self.head_index = True
        self.mem_data = pd.DataFrame()
        self.job_cols = []

    def _convert_type(self, x, out):
        job_counts = pd.DataFrame(x)
        job_counts["Status"] = job_counts["Status"].map({"Aborted": "A", "Finished": "F", "Scheduled": "S", "Cancelled": "C"})
        # job_counts = job_counts[~job_counts["Status"].isin(["A", "Aborted"])]  # almost always 0
        job_counts = job_counts[job_counts["Status"] == "F"]  # TODO
        for col in self.job_cols:
            if col in job_counts.columns:
                job_counts[col] = job_counts[col].astype(int)
        job_counts.to_csv(out, mode="a", header=self.head_index, index=False)
        self.head_index = False

    def _reindex(self, x, out):
        data = pd.DataFrame(x)
        data = data.set_index(['SysID', 'Date'])
        if self.head_index:
            self.mem_data = data
        else:
            self.mem_data = self.mem_data.append(data)
        self.head_index = False

    def _join(self, y):
        df2 = pd.merge(self.mem_data, y, how='inner', left_index=True, right_index=True)
        df2.to_csv("memory_usage_tmp.csv", mode="a", header=self.head_index, index=True)
        self.head_index = False

    def convert_pickle(self):
        f = open('../mem_data.pickle', 'rb')
        data = pickle.load(f)
        mem_data = pd.DataFrame(data)
        mem_data.to_csv('mem_data.csv', index=False)
        print("Finished exporting mem data.")

        del mem_data

        f = open('../job_cols.pickle', 'rb')
        data = pickle.load(f)
        self.job_cols = data[1:]
        print("Tracked jobs count:")
        print(len(self.job_cols))

        f = open('../job_counts.pickle', 'rb')
        data = pickle.load(f)
        job_counts = pd.DataFrame(data)
        job_counts.to_csv('job_counts.csv', index=False)
        print("Finished exporting job counts.")

        # creating a empty bucket to save result
        mem_data = pd.read_csv('mem_data.csv')
        df_result = pd.DataFrame(columns=(mem_data.columns.append(job_counts.columns)).unique())
        df_result.to_csv("memory_usage_tmp.csv", index_label=False)

    def preprocess(self):

        reader = pd.read_csv("job_counts.csv", chunksize=10000)  # chunksize depends with you colsize
        self.head_index = True
        [self._convert_type(r, "job_counts_type.csv") for r in reader]
        print("Finished converting job counts.")

        # reader = pd.read_csv("mem_data.csv", chunksize=50000)  # chunksize depends with you colsize
        # self.head_index = True
        # [self._reindex(r, "mem_data_reindex.csv") for r in reader]

        self.mem_data = pd.read_csv('mem_data.csv', index_col=['SysID', 'Date'])
        print("Finished reading memory data.")

        reader = pd.read_csv("job_counts_type.csv", index_col=['SysID', 'Date'], chunksize=10000)  # chunksize depends with you colsize
        self.head_index = True
        [self._join(r) for r in reader]
        print("Finished joining job count data.")

        with open("memory_usage_tmp.csv", 'r') as f:
            with open("memory_usage.csv", 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)

        '''
        memory_usage = pd.read_csv('memory_usage_tmp.csv')
        memory_usage.sort_values(by=["Date", "SysID"])
        memory_usage.to_csv("memory_usage_tmp.csv", index_label=True)
        print("Finished sorting memory usage data.")
        '''


if __name__ == '__main__':
    # preprocess = PreprocessData()
    # preprocess.convert_pickle()
    # preprocess.preprocess()

    mem_data = pd.read_csv('memory_usage_finished.csv')
    mem_data = mem_data.loc[:, (mem_data != 0).any(axis=0)]
    mem_data['index'] = mem_data['Date']
    first_col = mem_data.pop('index')
    mem_data.insert(0, '', first_col)
    mem_data['id'] = mem_data['SysID']
    sys_counts = mem_data['id'].value_counts()

    dates = mem_data['Date'].unique()
    systems = mem_data['id'].unique()
    index = pd.MultiIndex.from_product([dates, systems], names = ["Date", "id"])
    cross_join = pd.DataFrame(index=index).reset_index()
    merged = pd.DataFrame()

    for system in systems:
        df1 = cross_join[cross_join['id'] == system]
        df2 = mem_data[mem_data['id'] == system]
        merged_res = df1.merge(df2, left_on=["Date", "id"], right_on=["Date", "id"], how='left')
        merged = merged.append(merged_res, ignore_index=True)

    # merged = cross_join.merge(mem_data, left_on=["Date", "id"], right_on=["Date", "id"], how='left')
    print(merged)
    print(len(merged))
    print(len(mem_data))
    print(cross_join)
    print(mem_data['id'].unique())
    print(len(mem_data['id'].unique()))
    print(sys_counts)
    print(len(sys_counts))
    print(len(sys_counts[sys_counts == 221])) # pad 887 systems
    # start_date = date(2019, 7, 27)
    # start_date = date('2019-07-27')
    start_date = datetime.strptime('2019-07-27', '%Y-%m-%d')
    mem_data['days_from_start'] = mem_data['Date'].apply(lambda x : (datetime.strptime(x, '%Y-%m-%d') - start_date).days)
    mem_data['weekday'] = mem_data['Date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d').weekday())
    mem_data['month'] = mem_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    mem_data['year'] = mem_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').year)
    mem_data['day'] = mem_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)


    #mem_data.to_csv('memory_usage_finished_no_zero.csv', index=False)
    mem_data_test = mem_data[['', 'id', 'SysID', 'Date', 'Mem_avg', 'ActiveTsEntries_sum', 'days_from_start', 'weekday',
                              'month', 'year', 'day']]
    #mem_data_test.to_csv('memory_usage_finished_no_zero_test.csv', index=False)

