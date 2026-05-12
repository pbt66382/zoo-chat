# Phase 3 评估报告 - 意图分类

- 数据集大小：**61** 条
- 整体准确率：**100.00%**

## 各意图指标

| intent | total | tp | precision | recall | f1 |
|---|---|---|---|---|---|
| greet | 6 | 6 | 100.00% | 100.00% | 100.00% |
| meeting_create | 6 | 6 | 100.00% | 100.00% | 100.00% |
| meeting_join | 6 | 6 | 100.00% | 100.00% | 100.00% |
| screen_share | 6 | 6 | 100.00% | 100.00% | 100.00% |
| schedule_meeting | 6 | 6 | 100.00% | 100.00% | 100.00% |
| troubleshoot_audio | 6 | 6 | 100.00% | 100.00% | 100.00% |
| troubleshoot_video | 6 | 6 | 100.00% | 100.00% | 100.00% |
| troubleshoot_network | 6 | 6 | 100.00% | 100.00% | 100.00% |
| general_inquiry | 6 | 6 | 100.00% | 100.00% | 100.00% |
| out_of_scope | 7 | 7 | 100.00% | 100.00% | 100.00% |

## 混淆矩阵

行=实际意图，列=预测意图。

| actual \ pred | greet | meeting_create | meeting_join | screen_share | schedule_meeting | troubleshoot_audio | troubleshoot_video | troubleshoot_network | general_inquiry | out_of_scope |
|---|---|---|---|---|---|---|---|---|---|---|
| greet | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| meeting_create | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| meeting_join | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| screen_share | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| schedule_meeting | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| troubleshoot_audio | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| troubleshoot_video | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| troubleshoot_network | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| general_inquiry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| out_of_scope | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |

## 端到端评估

- 数据集大小：**30** 条
- Top-K：**3**
- 是否走 generation：**False**

- 意图准确率：**93.33%**
- Recall@3（召回包含期望 FAQ）：**100.00%**

### 明细

| question | expected_intent | got_intent | expected_faq | recalled | intent_hit | recall_hit | keyword_hit |
|---|---|---|---|---|---|---|---|
| 怎么发起一场视频会议 | meeting_create | meeting_create | 1 | 1,10,9 | OK | OK | - |
| 我要立即开个会 | meeting_create | meeting_create | 1 | 1,4,9 | OK | OK | - |
| 怎么加入别人发的会议 | meeting_join | meeting_join | 2 | 2,9,112 | OK | OK | - |
| 邀请链接怎么用来入会 | meeting_join | meeting_join | 2 | 2,9,112 | OK | OK | - |
| 怎么共享屏幕 | screen_share | screen_share | 3 | 3,79,59 | OK | OK | - |
| 怎么把我的桌面分享出去 | screen_share | screen_share | 3 | 3,15,59 | OK | OK | - |
| 我想预约下周一上午十点的会议 | schedule_meeting | schedule_meeting | 4 | 4,65,19 | OK | OK | - |
| 怎么提前安排好一次会议 | schedule_meeting | schedule_meeting | 4 | 89,4,148 | OK | OK | - |
| 会议中我听不到声音 | troubleshoot_audio | troubleshoot_audio | 5 | 5,21,6 | OK | OK | - |
| 对方听不到我说话怎么办 | troubleshoot_audio | troubleshoot_audio | 6 | 6,5,44 | OK | OK | - |
| 怎么打开摄像头 | troubleshoot_video | troubleshoot_video | 7 | 7,140,48 | OK | OK | - |
| 视频画面太模糊了 | troubleshoot_video | troubleshoot_video | 8 | 8,73,145 | OK | OK | - |
| 怎么邀请同事参加 | meeting_join | meeting_join | 9 | 9,2,87 | OK | OK | - |
| 怎么录制会议视频 | general_inquiry | general_inquiry | 10 | 10,64,50 | OK | OK | - |
| 之前录的会议视频在哪里 | general_inquiry | general_inquiry | 11 | 11,10,34 | OK | OK | - |
| 会议怎么设密码 | general_inquiry | schedule_meeting | 12 | 12,70,43 | X | OK | - |
| 怎么开启等候室 | general_inquiry | general_inquiry | 13 | 13,75,35 | OK | OK | - |
| 怎么切换发言人视图 | general_inquiry | general_inquiry | 14 | 14,74,50 | OK | OK | - |
| 想用虚拟背景遮住房间 | troubleshoot_video | troubleshoot_video | 17 | 17,77,85 | OK | OK | - |
| 网络断了会议怎么办 | troubleshoot_network | troubleshoot_network | 18 | 18,58,95 | OK | OK | - |
| 会议中老掉线怎么解决 | troubleshoot_network | troubleshoot_network | 18 | 18,58,95 | OK | OK | - |
| 怎么关闭别人的麦克风 | general_inquiry | troubleshoot_audio | 44 | 44,6,60 | X | OK | - |
| 会议结束按钮在哪 | general_inquiry | general_inquiry | 47 | 47,20,93 | OK | OK | - |
| 怎么打开实时字幕 | general_inquiry | general_inquiry | 49 | 49,84,50 | OK | OK | - |
| 怎么开启自动转录 | general_inquiry | general_inquiry | 50 | 84,50,49 | OK | OK | - |
| 我想忘记密码了怎么找回 | general_inquiry | general_inquiry | 30 | 30,70,12 | OK | OK | - |
| 怎么修改我的头像 | general_inquiry | general_inquiry | 27 | 27,37,28 | OK | OK | - |
| 会议有人数上限吗 | general_inquiry | general_inquiry | 105 | 105,42,46 | OK | OK | - |
| 想升级套餐 | general_inquiry | general_inquiry | 117 | 117,46,119 | OK | OK | - |
| 怎么联系人工客服 | general_inquiry | general_inquiry | 122 | 122,120,98 | OK | OK | - |
