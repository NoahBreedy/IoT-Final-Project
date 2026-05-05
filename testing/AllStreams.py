from StreamData import StreamData

ALL_STREAMS:dict[StreamData] = {
    # "CamFull": StreamData("http://localhost:8080/stream/0"),
    # "CamBL": StreamData("http://localhost:8080/stream/1"),
    # "CamTL": StreamData("http://localhost:8080/stream/2"),
    # "CamBR": StreamData("http://localhost:8080/stream/3"),
    # "CamTR": StreamData("http://localhost:8080/stream/4"),
    "0": StreamData("http://172.20.10.9:81/stream"),
    # "1": StreamData("http://172.20.10.8:81/stream"),
    "2": StreamData("http://172.20.10.7:81/stream"),
}