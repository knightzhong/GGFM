UAB-TD/

│

├── main.py                # 项目入口：串联所有流程

├── requirements.txt       # 依赖包

│

├── src/

│   ├── __init__.py

│   ├── config.py          # 超参数配置（集中管理）

│   ├── utils.py           # 工具类：标准化、随机种子、设备管理

│   ├── oracle.py          # 阶段一：NTK/GP 安全教师模型

│   ├── generator.py       # 阶段一：轨迹生成与筛选 (Distillation)

│   ├── models.py          # 阶段三：Flow Matching 神经网络架构

│   └── flow.py            # 阶段三&四：流匹配训练 Loss 与 ODE 推理求解器
