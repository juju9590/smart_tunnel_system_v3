# ==========================================
# 파일명: cctv_whitelist.py
# 위치: backend_flask/modules/tunnel/cctv_whitelist.py
# 역할:
# - 테스트용 CCTV 화이트리스트 관리
# ==========================================

TEST_CCTV_LIST = [
    {
        "name": "[수도권제1순환선] 사패터널(퇴계원방향)",
        "url": "http://cctvsec.ktict.co.kr/5008/LG8ZJslapF3Rx6x3km0zycjOvkMw6vEpw2rZsY1aXBNqzvQqP7lDWzXX0AouU+FxN4ymm+kuoNxbN/sq5FkUF3R8wC6/jnWqJMCBCRe3kIU=",
    },
    {
        "name": "[수도권제1순환선] 사패터널(일산방향)",
        "url": "http://cctvsec.ktict.co.kr/5009/BNNvpG0Y2Yt4c+e/jjy1bGs5AuzI/P5VeGSCmSjJ01vSVAk75pkGVSWwsgrsXVsnbGsxu6L7CFCZlyg6JXpseqUCJKLUOc5Ft/JNatdubog=",
    },
    {
        "name": "[서울양양선] [양양]화촌9터널(양양5)",
        "url": "http://cctvsec.ktict.co.kr/3605/VW5rlIG7NmaLcF2DD5tmS0qK+SIWUj8UU0p9IbhTr4UAeZGBCaOyep/vUVNJSlZYNQu2fJoPbYRlnmdg3Zea2+NiegiB5ClRbYvuwTSchAw=",
    },
    {
        "name": "[서울양양선] [서울]화촌9터널(서울12)",
        "url": "http://cctvsec.ktict.co.kr/3606/sndIwISVn1wVhqTTjUmHiWTqURI1bNXfoSkmre7hub1/VJIaTk/Q5+P60ZaKFHmzsjMbMRpIsPtBgknTrHBRYyi4812ViPAshEZAceidsJc=",
    },
    {
        "name": "[서울양양선] [양양]화촌9터널(양양8)",
        "url": "http://cctvsec.ktict.co.kr/3607/fRsu8pF3tAMeggMwpvpoi8bC39GtUBw5K6hHy2xOA6MPMh9jYqLT5QijAHLzhoTpnrN+H1pIIyQZ5y/d5xQJPwyc0Z1voCzwIXVTspxC1NM=",
    },
    {
        "name": "[서울양양선] [서울]화촌9터널(서울8)",
        "url": "http://cctvsec.ktict.co.kr/3608/KVUqlvRFDI0SQynOps7ueIZvpRKGccWH2g+ih/zbuulskfuOl/t0aHi+iLOEUXeAmvVu+E6c2W2So2z7CYsaaQL+ZwJOKjtCzTRd8fm1KWs=",
    },
    {
        "name": "[서울양양선] [양양]화촌9터널(양양11)",
        "url": "http://cctvsec.ktict.co.kr/3609/+C2qP79tg8uN+LDk3kNQ8qI0RkYVamEYXi/txSNIpWIsdPALks83Vhgj2Hg13WYJ6XULCIAPx3prBqWNZj5Nof+YwQT6+ZDFDzV2rHcP5tA=",
    },
    {
        "name": "[서울양양선] [서울]화촌9터널(서울5)",
        "url": "http://cctvsec.ktict.co.kr/3610/IUtI57QTBrHJOMvDCUzDJnaDO5+jxLUtKrXzIV34VYv3NxgAWuUCbcdErjJL9WAfu/envEnLzzBR64AYi2kwJYKGWHHID2qefVii75BN8/I=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(116/116)",
        "url": "http://cctvsec.ktict.co.kr/3652/mQM3TsgBxO7+oSK4giC2rPnEkwbHN6M8q3W/6vukBWlzhl4a/YHiyx6eCZ6aquJuxFr1PxdFoJx2ulD1hEIG8QphRfGHzcUo+LP7qu711Qg=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(97/116)",
        "url": "http://cctvsec.ktict.co.kr/3653/6wu+UjBZoGHLuRyGFdeaurm9Cn7bGiMKIno2XUaZ6BqSN/06oUu4+urcns7GPswdI3OdwEbMoYa1PxqKI2juNwsXF3jgMkjR58nSexRCPqQ=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(23/115)",
        "url": "http://cctvsec.ktict.co.kr/3654/4FqS+Yb2h1FDgvzRKTpOjugvMk4PLEPjz4MAT4W2b+KSF0BMmQIcaTBRbbPrDQtpW2ZpJcdJ9OcBhnwGpX7I2UH8Z36Ep09s8AmDxl35nqQ=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(32/115)",
        "url": "http://cctvsec.ktict.co.kr/3655/vfj8nAXw1K3zMl2bAxeSFtEOAMiIJHZaOKHRNla+1tBDYJ/NSiBs0uW39ywOZms/ZelejcUPgSr3eqImKTgRyXhbyakCJBa2KrtQiokLo50=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(86/116)",
        "url": "http://cctvsec.ktict.co.kr/3656/jfppO9dmRH/1W3NioTQ+OLV2u0yBkB7N2olhaa60PdbvywXwGHJpE5A/YAfMb0uGGYl9+wQlsJ+A+AcGhz1LoZ5yrsnVf8uJ7wh+a/I24Tg=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(44/115)",
        "url": "http://cctvsec.ktict.co.kr/3657/C4NCbjb7B5+gjUvPNEaIKvdFjeTYVuXJ2oaDo9CZBm2Lf2p9k8Im0gkIDD35X0kr6hFdw6C9P4L6jBY0G6ukF60BZC9Pb+5gNwRN7x54js0=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(75/116)",
        "url": "http://cctvsec.ktict.co.kr/3658/M5a9twMD68z205yz8aO255LmZhBdpplorqnbgCzOMtIhoPBRoBiRA/NbW4wF+6DrMjGkxmMLt4gMLj1tK1jLLPZmGcJ/cY9rTnS4zfh/c9k=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(56/116)",
        "url": "http://cctvsec.ktict.co.kr/3659/OifCH9NiDTuj4WzXG6VP8bym8bh9oLJcGpTTRk9U/njsnJVWzV9Jio6msvhNGaamClL5Z75GS4YoIHdrqWvu9/LC51nENisogETMIKifq88=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(63/115)",
        "url": "http://cctvsec.ktict.co.kr/3660/UWvdHbjp9C9+2jTsKrJ6QCfta2tQKW7WFbtdFIRF7SiFsQKw/V5DucPtOGWc18L5qGdT4bUmJ+FOqVctwFOcbrQz2eubXlWqJ5YwD7v5y+Q=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(44/116)",
        "url": "http://cctvsec.ktict.co.kr/3661/+iQQsGY86fSI0IFlPr2Nil3Ecq0TlXzyFXWCLboHUlAr/O3+ehhDdSE3Ks9fUU+uJ+kdR12m+v7hw9ySvZIqIWfT+EZ7S+EbV29I8aBVJY0=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(75/115)",
        "url": "http://cctvsec.ktict.co.kr/3662/uNQ5qT7kyPNxBx/ILVnvi044lKbEU5UhE8aVdS7WzZuFgoY0FxjucOAbb5Pfc4jvBfPVKrbj8ubiRvdzkKOUALFa9SVJWyAf/dQoPUv5Hzo=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(25/116)",
        "url": "http://cctvsec.ktict.co.kr/3663/mxh5XUs21jdvF24AhGze3D8amr4GLR7USKUuz/qM5U0D+aIKvE2CWqC0XROtiABNCemjZxrAZQkj0TpzbO2H/GkpcPXw9eQG3nz2RxQxk5E=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(95/115)",
        "url": "http://cctvsec.ktict.co.kr/3664/KTGpfBWVXTDPg78xMSTqDBx+Ih/gYg5J+0aPMOcHLHGUxS+Lpvp7xrzxtYHfg3l+Pkyu3BdkSGZcUzi28CLPri/eofR5MhUzgvl6twHlpcc=",
    },
    {
        "name": "[서울양양선] [양양]인제양양터널(106/115)",
        "url": "http://cctvsec.ktict.co.kr/3665/diSXqJ97I4RqXCbIgrsiCrMk0Plzvqs2UmFVPJYIugFays7u39o8VTWxpU8S8dr0mHJcMDTDYPAmVJEHaPRwJMhXCLVhKrCzAgvF4y7UKo4=",
    },
    {
        "name": "[서울양양선] [서울]인제양양터널(13/116)",
        "url": "http://cctvsec.ktict.co.kr/3666/6eGZwBAH0hf0Zxi4Vj9jXwVfJ0D7JtaBEapBA8GMsgDJMMk9mbom65bq9FK2cqZvrR5YXZoI7XTKWh5RDOvK6tEOK+NqkBqHS5JDOhW3PfM=",
    },
    {
        "name": "[서울양양선] [서울]화촌3터널(서울2)",
        "url": "http://cctvsec.ktict.co.kr/3592/Fz/eBy5gRIKB9CrxbKWy3k5gWL50GpwwcrKtUwiIBHPyW77VhO62A1pSkX3C5LKmsjhrwgGADQULc9Cy7DwvqGWs4W/MeeigdX+15mPQgmk=",
    },
    {
        "name": "[서울양양선] [양양]화촌3터널(양양2)",
        "url": "http://cctvsec.ktict.co.kr/3593/IPN0s8c5HKtMXqYeYMsnpWCGnIBW87kBfNq1MFAyAPXoMQjs3Tgi73IJ/ELz1S7n8bhdeqqGAIfHRNEr+bvbf3mgi6irmVmwnzsMZBBv/nk=",
    },
    {
        "name": "[서울양양선] 화촌3터널",
        "url": "http://cctvsec.ktict.co.kr/3594/Pbvr+GsVeqJ3gQ6E1tinXwYu+f2+Iv7uXkwaLS6B+VMpRu/oDwjy+S+CgQDqhgYJY67EKjI9SovnrTQgidKtb6VYIdYR0zIa+/b2s6QHzws=",
    },
    {
        "name": "[서울양양선] 화촌6터널",
        "url": "http://cctvsec.ktict.co.kr/3597/VKwwz6n7dI4Aq4Y9KMUaPEsg4KXbaAnlxsQ0hFhCG1/oX9S2nJe0q0r8LRoiht41IvQTtA5++Sy8UI73SHqHF2C2+htHXCvM6+nwCuVpzY0=",
    },
    {
        "name": "[서울양양선] [서울]화촌6터널(서울3)",
        "url": "http://cctvsec.ktict.co.kr/3598/JyEbvrXsiijIkb5FZc0sewke8mfyZHu3zPuejTox0pY6VC2GnrVLr6prFEVaoGHgUuGqnA5ru2kTV70FkkyyUdOazwF0VloMzYuehX7yyBg=",
    },
    {
        "name": "[서울양양선] [양양]화촌6터널(양양3)",
        "url": "http://cctvsec.ktict.co.kr/3599/0ODIrT/p5DifTD6pYshtO2mgaDfAK7m7KFkPU8pTCn1Ml1N7H5FpEzncu+KjAjeInAhFD7O92unwxNac9G8aehfW1WBPmH06CVUewvaxnl4=",
    },
    {
        "name": "[영동선] 광교터널",
        "url": "http://cctvsec.ktict.co.kr/58/4JYxkW8Wz4hLvF5EjTdlHvEY5+hNwHnlWr8ygQhliFIPRk2PODwC2D06Mex5cC2ruiOn+vQ2UK58YfEzkcTEZQW8m+uR3nbMrqVGrptMmhQ=",
    },
    {
        "name": "[영동선] 마성터널(인천)",
        "url": "http://cctvsec.ktict.co.kr/83/+2ZbZkQGWvgx7SYnUMeuXKYgG9X9Dg88gwG0GXzAraNATrLbR/0UNnFzu8Cr6lfmruvpFBvtq6ZuZeYk5bibDMbA8VlzxbZTPsjEZvPPNLs=",
    },
    {
        "name": "[영동선] 양지터널(강릉)",
        "url": "http://cctvsec.ktict.co.kr/85//SmCJGqqvnaG1/aHqXNoHkfyvy3ciRIQharjNWL50eH/cbEiBx6M/Mjsg5qg6Sw7NmTj9pbieYYnbtkop5GyfbJe3/mLkL2rZ5bFBzrKCJA=",
    },
    {
        "name": "[평택제천선] 평택터널",
        "url": "http://cctvsec.ktict.co.kr/111/U2F9Fpfi2M7RxRx7Eou4+HbJs3ze6V3WLZQ3FQ/l/1JFinsWnhHrC6UW/QaFA2MuK9OI6V+DMkkjHaogUBrqv32HiBDIEWuUNEuBFowBgUc=",
    },
    {
        "name": "[수도권제1순환선] 불암터널(일산방향)",
        "url": "http://cctvsec.ktict.co.kr/5003/abRLXXkyGEleVGXcvO3+RLiSrgCxVofjXEUXH86sP+SfJ54eGg+mQGmPVfsACqXK+5dfeEiiFpGgk3tRdPFySW4PCbeWjp6gg2lmi0y6P0Y=",
    },
    {
        "name": "[수도권제1순환선] 불암터널(퇴계원방향)",
        "url": "http://cctvsec.ktict.co.kr/5004/zPcyVFGwqhaSZ9rc6emYnrlyCB6cW9I6SXwzWuH6CIO/hJv3yCFQZ75i39ToLUC+kiWJgv6HGVKN7gLoXgDasTgYnYQrszO718oA0Wd3nE8=",
    },
    {
        "name": "[수도권제1순환선] 수락터널(일산방향)",
        "url": "http://cctvsec.ktict.co.kr/5005/BwVYULim2VdKbi+Cu+O5xC0jd9kkspYitDzmgVMqlf4+axp1jZBuLrANWiS/4yK9Pa9k/IveqhADV3xZbtIsDCGEpbwWfm5MADCl9vnas8Y=",
    },
    {
        "name": "[수도권제1순환선] 수락터널(퇴계원방향)",
        "url": "http://cctvsec.ktict.co.kr/5006/B/vOHt7c/CNOwHuQYM3CmWTva6jw7SkFvXlhs/E7xA51QBAcCmiNTj2TMV4gCdfX9BEAF/iAg+eZ60IkO/RtteW2vF0E3Mt/wsXpwdeiUGI=",
    },
    {
        "name": "[수도권제1순환선] 사패터널(퇴계원방향)",
        "url": "http://cctvsec.ktict.co.kr/5008/LG8ZJslapF3Rx6x3km0zycjOvkMw6vEpw2rZsY1aXBNqzvQqP7lDWzXX0AouU+FxN4ymm+kuoNxbN/sq5FkUF3R8wC6/jnWqJMCBCRe3kIU=",
    },
    {
        "name": "[수도권제1순환선] 사패터널(일산방향)",
        "url": "http://cctvsec.ktict.co.kr/5009/BNNvpG0Y2Yt4c+e/jjy1bGs5AuzI/P5VeGSCmSjJ01vSVAk75pkGVSWwsgrsXVsnbGsxu6L7CFCZlyg6JXpseqUCJKLUOc5Ft/JNatdubog=",
    },
    {
        "name": "[안양성남선] 삼성산터널(성남)",
        "url": "http://cctvsec.ktict.co.kr/5460/HQJ9tSpfnVHgvLGVQKy7iOC2ocixpP14/cD69kmkJaRsPauOCLWJyBsiGZGhZBWXWwBq/9koloIFc74HmPrz2Mt7HC1PM5h32awjlr/F4n0=",
    },
    {
        "name": "[안양성남선] 삼성산터널(안양)",
        "url": "http://cctvsec.ktict.co.kr/5461/nOo+6RM8OZeP+mlYuYShMqzXbi+SWMlvV1sFLBxBtqWc5HwMqRtVAzrcAIQKkO8ZYTj0deUSsqrLIiWYBJPZckSEpE6UkhA/jLw4473KZBw=",
    },
    {
        "name": "[안양성남선] 삼성산터널(성남)",
        "url": "http://cctvsec.ktict.co.kr/5462/PYSudDoqlWoSi8BA4otT9Q5iF5MJP3eArYiGgEwx9wDnBSx1USq6E//nf1b3vJNJdyZ4oU2jTuGWnIqMy6Q0T95brAGtwoXJTeADVExMs44=",
    },
    {
        "name": "[안양성남선] 청계산3터널(안양)",
        "url": "http://cctvsec.ktict.co.kr/5467/Tjp7bPi81Pqpyhphyazu9HUQtxGVPXMr7udTl4orOAvdt984hqqWnHsxhlaUx86AkM/Ax9Rg2vzKc40U52sJvsd4mH6d0azpRxMoKGefdvU=",
    },
    {
        "name": "[안양성남선] 청계산3터널(성남)",
        "url": "http://cctvsec.ktict.co.kr/5468/DW0C2dNsIwnZzVkfaHjIRzFbkl/YmZKjFkSE4X43FmaBpQOaF7QVOsZww5qh3MqaF+bAdHO65gcW9sUX7QbGAFSbI2n3873VAxfVMHEYJEw=",
    },
    {
        "name": "[안양성남선] 청계산3터널(안양)",
        "url": "http://cctvsec.ktict.co.kr/5469/GBnhpON5URXjDbmFPHuiyHFEFt+fjrwiipvGcG1px72ewOm5KvZNqf8HN42LYLAuOiS4ujXT56NiQzd6iRUlvEI+3H+3tSoEOz9aBC8Z99s=",
    },
    {
        "name": "[안양성남선] 청계산3터널(성남)",
        "url": "http://cctvsec.ktict.co.kr/5470/Zj3W/dnJEywZhCd0Qqj2S06TAgXythf8EsPzAtwWHLVuJl2w9KkN7Pn2coC0GWWypjwFLV8NcizdcpyENEzgFVKv3GVQ8VywtkZuTIT3WBw=",
    },
    {
        "name": "[안양성남선] 청계산4터널(성남)",
        "url": "http://cctvsec.ktict.co.kr/5472/T6/aJs0f9ArcBHo2KoKqzEm3vkB9+VoBmyJpmHjrnlgtyxrdPFxxjxcctf10UAmC19e6JrX6F1ixYfqJzuuJBA2D9vHxH++Tfhe4BywMJm4=",
    },
    {
        "name": "[부산외곽선] [창원]금정산터널(창원15고정)",
        "url": "http://cctvsec.ktict.co.kr/301/FYN5ow5Ym+Z+4XYLi1yRRh0tD/1Kp0pSiW0eFTjgMr2gfWOmDNUYlhioPBbOf8ZPMpm9GzY4+WMW7C5UPKk9iWvZbsdJCQECxyC+lq6LThQ=",
    },
    {
        "name": "[부산외곽선] [기장]금정산터널(기장22고정)",
        "url": "http://cctvsec.ktict.co.kr/302/9zLjf19Cw38L0WcZ+kGIEDDkXvp9/4TlUKT7tgaWGAjyZA7JWiEGWiz71vRBsjcFCQ7+J4QE6wpC2Z7uQ0n3OkSK719emyg6GfiZ7Pb2xwQ=",
    },
    {
        "name": "[남해선] 김해터널",
        "url": "http://cctvsec.ktict.co.kr/2521/zmvKLtMyiHrgbwD9YtcOrPsfTfh10Fj8QqbbOTxB5uw8vo0vEE8VkwxIBjr3XPWJMDx5hPRiRGWQItDA94V3b2gt0Pt/rkW9t2M55BdrisI=",
    },
    {
        "name": "[수원광명선] 구봉산터널입구_광명방향",
        "url": "http://cctvsec.ktict.co.kr/5175/W+xzod6pUCO+GOjryJwT6ZopJqZ7+ZG8+0XZe1rxM798zewh1APmKT1kkw2Xax2NNBqXux1qHxd51R0U6rSHjcVrzEpWbumXUPntsq4dFxo=",
    },
    {
        "name": "[수원광명선] 광명 구봉산터널",
        "url": "http://cctvsec.ktict.co.kr/5176/kbe5SBsTXBbX0i4hdDuuSE5ZilAFwOQmPbMJch63jW/B6gwf4akV/GlpTDH8JL4t/G5lf7MncT+kRWOa3OYBqw6Z3vofjYfuMlcSlyaEZOM=",
    },
    {
        "name": "[수원광명선] 수원 구봉산터널",
        "url": "http://cctvsec.ktict.co.kr/5177/L9EzbfGXilhFTE5N63a8MH4UBqXrd/O8w9zGEaMHxxXrSB8CltpTYSaeQyPLaD56cv5laU25qOVvX/lysxfETzs2eLrDnmQgOnI9h6OcJD8=",
    },
    {
        "name": "[수원광명선] 구봉산터널입구_수원방향",
        "url": "http://cctvsec.ktict.co.kr/5178/W+IJ6EXoTAA2dCXANnRUaokOcVL7l3tXdLpGzS7U0G1Ps3Lo5YZA5135mPt4RsQ9LJg9egJgw+z1kDrrm0Bb0RUveJvkT6YpYI5lH+PXeLk=",
    },
    {
        "name": "[수도권제2순환선(봉담동탄)] 필봉산터널(봉담)",
        "url": "http://cctvsec.ktict.co.kr/8326/tD+wiAbI/YfgrvESpV516dvSLQqee4qTAwt2mAYU6ROZ7nh6OHstwhRYnwIwYTgEBu1iZT4VTFyIa6Vwg+cVI3dUhm82yl2i+tDPTipVH3E=",
    },
    {
        "name": "[수도권제2순환선(봉담동탄)] 필봉산터널(동탄)",
        "url": "http://cctvsec.ktict.co.kr/8327/JdqIr+tRRcj6oVvFQnzIvtzHkUDVhJZY9XY+eGNGobDo55Y5O5qOjHvT6ff9uRudnQ7jtswETvv+M/fs9ia+cqNyXt2YgXEmO36dKnPo3bg=",
    },
]