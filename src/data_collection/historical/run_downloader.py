from src.data_collection.historical.historical_aq import HistoricalDownloader

downloader = HistoricalDownloader()

downloader.download_all(23747)