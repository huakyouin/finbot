## demo说明

后端依赖：

```bash
pip install Flask Flask-Cors
```

前端依赖：

```
###########################
## 如有npm可跳过本段
###########################
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
chmod +x ~/.nvm/nvm.sh
source ~/.bashrc
nvm install 18
npm install -g nrm
nrm add taobao https://registry.npm.taobao.org
nrm use taobao
###########################
cd frontend
npm install
```

然后用两个终端分别启动前后端：

```bash
cd frontend
npm run dev
```

```bash
cd backend
python app.py
```