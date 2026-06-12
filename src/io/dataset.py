import pandas as pd
import glob


class Files:
    def __init__(self, location: str):
        self.location = location

    def get_files(self) -> list:
        self.files = glob.glob(self.location)
        self.total_files = len(self.files)
        return self.files


class Dataset:
    def __init__(self, files: Files):
        self.files = files

    def loadData(self):
        files = self.files.get_files()
        if len(files) == 0:
            return None
        self.columns = pd.read_csv(files[0]).columns
        self.data = pd.DataFrame(columns=self.columns)
        for f in files[:10]:
            name = f.split("/")[-1].split("__")[0]
            temp = pd.read_csv(f)
            temp["name"] = name
            self.data = pd.concat([self.data, temp], axis=0)
        return self.data

    def processData(self, processor):
        self.processed_data = pd.DataFrame()
        for file in self.files.get_files()[:10]:
            file_name = file.split("/")[-1].split("__")[0]
            p_data = processor.transform(pd.read_csv(file))
            # p_data["name"] = file_name
            self.processed_data = pd.concat([self.processed_data, p_data], axis=0)
        return self.processed_data
