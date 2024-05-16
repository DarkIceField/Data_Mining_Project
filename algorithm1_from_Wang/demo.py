# encoding:utf-8
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt


def radar_chart(traits:dict):
    # 数据
    labels = list(traits.keys())
    data = list(traits.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))  # 闭合雷达图，将首尾元素连接起来
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='skyblue', alpha=0.6)
    ax.plot(angles, data, color='blue', linewidth=2, linestyle='solid')
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    # 返回雷达图
    return fig


def predict_plant_traits(input_img):
    # 预测植物性状
    traits = {'X4':0.48, 'X11':15.75, 'X18':0.39, 'X26':0.64, 'X50':1.35, 'X3112':574.10}
    # 性状归一化
    traits_percentage = {'X4': 0.48, 'X11': 0.15, 'X18': 0.39, 'X26': 0.64, 'X50': 0.135, 'X3112': 0.57}
    return str(traits), radar_chart(traits_percentage)


demo = gr.Interface(
    fn=predict_plant_traits,
    inputs=gr.Image(),
    outputs=["text", "plot"],
    title="植物性状预测",
    description="上传图片，查看预测的植物性状。"
)
demo.launch()