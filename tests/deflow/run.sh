#!/bin/bash
# Download from https://github.com/kylevedder/zeroflow_weights/raw/master/argo/supervised/supervised.ckpt 
# and save as /tmp/fastflow3d.ckpt if it does not exist

if [ ! -f /tmp/deflow.ckpt ]; then
curl 'https://hkustconnect-my.sharepoint.com/personal/qzhangcb_connect_ust_hk/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fqzhangcb%5Fconnect%5Fust%5Fhk%2FDocuments%2FPublic%5FShared%5FOnline%2FPre%2Dtrain%20weights%2FDeFlow%2Fdeflow%5Fofficial%2Eckpt' \
  -H 'authority: hkustconnect-my.sharepoint.com' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'cookie: MicrosoftApplicationsTelemetryDeviceId=0638f4fc-fad5-4fd3-8a52-85bfe7aeffda; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2MzMDNlNGU2NWFiNzkxNTMzNjg2ZjNlMmQ2OGQxYjkxMmZkOTBlNTU2ODkyNTdmNmZmNzNiYTg2ODIzYzY1OWEsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jYzMwM2U0ZTY1YWI3OTE1MzM2ODZmM2UyZDY4ZDFiOTEyZmQ5MGU1NTY4OTI1N2Y2ZmY3M2JhODY4MjNjNjU5YSwxMzM1Njk5OTg4MTAwMDAwMDAsMCwxMzM1NzA4NTk4MTI4NzE2NTksMC4wLjAuMCwyNTgsMjNiYWFmYjQtNDllZC00M2EyLTlmNTItYThhMzE3YTg1ZDhkLCwsZDhjYTFjYTEtMDBjZi0zMDAwLTI0M2ItNDM5YWQzM2FiMzg4LGQ4Y2ExY2ExLTAwY2YtMzAwMC0yNDNiLTQzOWFkMzNhYjM4OCxJSFIvWk0vUm5rNk5kSStpT0gxQnBBLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxOTYwMTAsRGFEQWZqUVFtcHlPWHgyUnJLX1c1bHZvTFo0LFQzaEV3WHRXSzEvY1ZFOW1xZWRwdHl0V3M3cno1Nm9WMFZ5dXQrUmlxK0dGNUVRbmZvRFdqYitseDdwdmp1VU5JWkRpZkovR3FrZmFWNnFoVG03WmVKSTdDd1ZlVmNFU1V6K1dHRklVUml4V1FSZTN3b3R2dW5sV1NVMjE0MlhGNmJkVkwydE1Fd0twTFFpdEVBelBxdU9yeldkczFBUldxcDN6ZUVFRTU5UmZaNkRWeVVFRVl0MzV5Vm8xUktwcFdENy9FdkxLdXVsME9lbk9IL0J2R2ZQWm9hZW91V2Jjd3lPR01semo5UTliNHlWeXZDMngzRC9LOVpUZHFxSWN2MnN0Q2QwMmEvSGppc0NQNGRFeUIwZHVpWDRJeXN3SmUwbUE5aGkxVmhHRHVOVHFSRFFsVTdLb2pTaGw0bzVXdmdMOWtQTHhpNk84MXo3NU1YSkZnZz09PC9TUD4=; ai_session=R4qPHlJWUpPdaW+ZDvsHJ1|1712526010920|1712526010922; MSFPC=GUID=111bdd0a2ded4dc5b536a2893f36fc81&HASH=111b&LV=202312&V=4&LU=1701999763540' \
  -H 'dnt: 1' \
  -H 'referer: https://hkustconnect-my.sharepoint.com/personal/qzhangcb_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fqzhangcb%5Fconnect%5Fust%5Fhk%2FDocuments%2FPublic%5FShared%5FOnline%2FPre%2Dtrain%20weights%2FDeFlow%2Fdeflow%5Fofficial%2Eckpt&parent=%2Fpersonal%2Fqzhangcb%5Fconnect%5Fust%5Fhk%2FDocuments%2FPublic%5FShared%5FOnline%2FPre%2Dtrain%20weights%2FDeFlow' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Linux"' \
  -H 'sec-fetch-dest: iframe' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: same-origin' \
  -H 'sec-fetch-user: ?1' \
  -H 'service-worker-navigation-preload: {"supportsFeatures":[1855,61313]}' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
    --compressed -o /tmp/deflow.ckpt
fi

pip install omegaconf

python test_pl.py tests/deflow/config.py --cpu --checkpoint /tmp/deflow.ckpt
