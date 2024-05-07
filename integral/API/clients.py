import http.client

conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")

headers = {
    'X-RapidAPI-Key': "a61b6ce860msha2aa06aed961d78p176ab6jsne549d703a2e1",
    'X-RapidAPI-Host': "api-football-v1.p.rapidapi.com"
}

conn.request("GET", "/v3/timezone", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))