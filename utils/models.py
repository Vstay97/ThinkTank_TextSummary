from torch import nn
from my_layers import MultiHeadSelfAttention
from torch.nn import functional as F

class My_Model(nn.Module):
    pass

def get_model(args,device):
    pass

def net(cur,prob):
    """
        cur.shape:
            'input_ids': text_ids,
            'decoder_input_ids': summary_ids,
            'attention_mask': [1] * len(text_ids),
            'decoder_attention_mask': [1] * len(summary_ids)

        prob == model(input_ids, decoder_input_ids, attention_mask, decoder_attention_mask)
        prob.shape == (batch_size,sentence_length,vocab_size) ==(4,13,50000)
    """
    # 获取摘要的 att_mask -> 获得mask掩码的矩阵
    # sum_mask.shape == (batch_size,sentence_length) == (4,13)
    sum_mask = cur['decoder_attention_mask']
    # 取句子本身的表达的mask。不要第一列,第一列是[cls]
    # sentence_mask.shape == (batch_size,sentence_length-1) == (4,12)
    sentence_mask = sum_mask[:,1:]
    # 将四个句子的mask全部铺平
    # mask.shape == (batch_size * sentence_length-1) == (48,)  == [1,1,1,.....,0,0]
    mask = sentence_mask.reshape(-1)
    # 转化为布尔值，作为最终的mask表达
    # mask.shape == (batch_size * sentence_length-1) == (48,) == [True,True,True,...,False,False]
    mask = mask.bool()

    ######################################   通过mask矩阵计算prob和labels   ###############################################

    # prob == model(input_ids, decoder_input_ids, attention_mask, decoder_attention_mask)
    # prob.shape == (batch_size,sentence_length,vocab_size) ==(4,13,50000)

    # 取句子本身的表达的token.不要最后一列,最后一列是[sep]
    # prob.shape == (4,12,50000)
    pro_three = prob[:, :-1]
    prob = prob[:, :-1]

    # 将prob重组成(x,50000)的格式,x为自我推断
    # prob_temp.shape == (batch_size * length-1,vocab_size) == (48,50000)
    prob_temp = prob.reshape(-1, 50000)
    # 将prob_temp用mask矩阵过滤
    # prob_temp_mask.shape == ( (batch_size*length-1)-len('False'), vocab_size ) == (44,50000)
    prob_temp_mask = prob_temp[mask]
    # 过滤后的矩阵为prob
    # prob.shape == (44,50000)
    prob = prob_temp_mask


    ##### 第一种：变换 (batch_size, emb_dim)
    ##
    # 加个线性层
    linear1 = nn.Linear(44,50000)
    activate = F.relu
    hidden = activate(linear1(prob))

    #### 第二种：变换 (batch_size, sentence_length, emb_dim)
    # 把pro_three的shape送进去
    mat = MultiHeadSelfAttention(4,12,50000)




    ########################################################################

    # 获得摘要的input_ids
    # summary_ids.shape == (batch_size,sentence_length) == (4,13)
    summary_ids = cur['decoder_input_ids']
    # 取句子本身的表达。不要第一列,第一列是[cls]
    # summary_exp.shape == (batch_size,sentence_length-1) ==(4,12)
    summary_exp = summary_ids[:, 1:]
    # 将矩阵打平
    # summary_pave.shape == ( batch_size * sentence_length-1 ) == (4*12,) == (48,)
    summary_pave = summary_exp.reshape(-1)
    # 将打平后的矩阵(列),用mask(列向量)过滤
    # 作为最终的labels表示
    labels = summary_pave[mask]


    # 返回预测值和标签
    return prob,labels

