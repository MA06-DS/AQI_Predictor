from src.data_collection.historical.historical_meteo import HistoricalWeatherDownloader
downloader = HistoricalWeatherDownloader()
downloader.download_all()