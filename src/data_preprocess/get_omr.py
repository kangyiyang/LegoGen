import requests
import os
from bs4 import BeautifulSoup
from tqdm import tqdm



def if_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_files(file_urls,path):
    file_paths=[]

    # 遍历URL列表，下载文件
    for url in tqdm(file_urls, desc="Downloading Files"):
        response = requests.get(url)
        if response.status_code == 200:
            # 从URL中提取文件名
            file_name = url.split("/")[-1].replace('%20',' ')
            
            file_path=os.path.join(path,file_name)
            file_paths.append(file_path)
            # 保存文件到本地
            with open(file_path, 'wb') as file:
                file.write(response.content)
        else:
            print(f'无法下载文件: {url}')
    
    return file_paths

def get_urls(urls_files):
    urls=[]

    for url_file in tqdm(urls_files, desc="Transfor to Url"):
        # 打开 HTML 文件
        with open(url_file, 'r',encoding='utf-8') as file:
            try:
                html_content = file.read()
            except:
                print(f'无法打开文件: {url_file}')
                continue
        # 解析 HTML 内容
        soup = BeautifulSoup(html_content, 'html.parser')
        # 查找<body>标签
        body_tag = soup.find('body')

        if body_tag:
            # 查找所有<script>标签
            script_tags = body_tag.find_all('script')

            # 逐个处理<script>标签内容
            for script_tag in script_tags:
                script_content = script_tag.get_text()  # 获取<script>标签内的文本内容
        else:
            print("<body>标签未找到")

        #观察发现下载地址在770字符位置
        index=770
        character=script_content[index]
        while character!='"':
            character=script_content[index]
            index+=1

        urls.append('https://omr.ldraw.org/'+script_content[770:index-1])

    return urls



path=r'Lego Dataset\omr'
urls_path=os.path.join(path,'urls')
models_path=os.path.join(path,'models')
if_exists(path)
if_exists(urls_path)
if_exists(models_path)


# URL列表，包含要下载的多个文件的URL
file_urls = []
for i in range(1,1838):
    file_urls.append(f'https://omr.ldraw.org/files/{i}')


# url_paths=[]
# files_and_folders = os.listdir(r'D:\Research\Lego Generation\Work\Lego Dataset\omr\urls')
# for i in files_and_folders:
#     url_paths.append(fr'D:\Research\Lego Generation\Work\Lego Dataset\omr\urls\{i}')

# 因为833和834下载的文件名一样，所以重复，导致少了一个文件
# for i in [833,834]:
#     url_paths.append(fr'D:\Research\Lego Generation\Work\Lego Dataset\omr\urls\{i}')


url_paths=download_files(file_urls,urls_path)
urls=get_urls(url_paths)
model_paths=download_files(urls,models_path)



