# AutoCrawler
Google, Naver multiprocess image crawler

![](img/animation.gif)

# How to use

1. Install Chrome

2. decompression chromedriver_xx.zip in chromedriver/

3. pip install -r requirements.txt

4. Write search keywords in keywords.txt

5. **Run auto_crawler.py**

6. Files will be downloaded to 'download' directory.


# Arguments
usage: python3 auto_crawler.py [--skip true] [--threads 4] [--google true] [--naver true]

--skip SKIP        Skips keyword already downloaded before. This is needed when re-downloading.

--threads THREADS  Number of threads to download.

--google GOOGLE    Download from google.com (boolean)

--naver NAVER      Download from naver.com (boolean)


# Data Imbalance Detection

Detects data imblance based on number of files.

When crawling ends, the message show you what directory has under 50% of average files.

I recommend you to remove those directories and re-download.

