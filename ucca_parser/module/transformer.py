import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class PositionEncoding(nn.Module):
    def __init__(self, word_dim, max_seq_len, padding_idx=0, freeze=True):
        super(PositionEncoding, self).__init__()
        n_position = max_seq_len + 1
        self.max_seq_len = max_seq_len
        self.freeze = freeze
        # self.position_enc = nn.Embedding.from_pretrained(
        #     self.get_sinusoid_encoding_table(
        #         n_position, word_dim, padding_idx=padding_idx
        #     ),
        #     freeze=freeze,
        # )
        self.position_enc = nn.Embedding(n_position, word_dim, padding_idx=padding_idx)
        nn.init.normal_(self.position_enc.weight)

    def forward(self, pos_idxs):
        return self.position_enc(pos_idxs)

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
        """ Sinusoid position encoding table """

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
        )

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.0

        return torch.FloatTensor(sinusoid_table)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill_((1 - mask).unsqueeze(1), -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, attention_drop=0.1, residual_drop=0.1, partition=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.partition = partition

        if not partition:
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

            nn.init.xavier_normal_(self.w_qs.weight)
            nn.init.xavier_normal_(self.w_ks.weight)
            nn.init.xavier_normal_(self.w_vs.weight)
        else:
            self.w_qs_c = nn.Linear(d_model//2, n_head * d_k //2, bias=False)
            self.w_ks_c = nn.Linear(d_model//2, n_head * d_k //2, bias=False)
            self.w_vs_c = nn.Linear(d_model//2, n_head * d_v //2, bias=False)
            self.w_qs_p = nn.Linear(d_model//2, n_head * d_k //2, bias=False)
            self.w_ks_p = nn.Linear(d_model//2, n_head * d_k //2, bias=False)
            self.w_vs_p = nn.Linear(d_model//2, n_head * d_v //2, bias=False)

            nn.init.xavier_normal_(self.w_qs_c.weight)
            nn.init.xavier_normal_(self.w_ks_c.weight)
            nn.init.xavier_normal_(self.w_vs_c.weight)
            nn.init.xavier_normal_(self.w_qs_p.weight)
            nn.init.xavier_normal_(self.w_ks_p.weight)
            nn.init.xavier_normal_(self.w_vs_p.weight)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=attention_drop)
        self.layer_norm = nn.LayerNorm(d_model)
        if not partition:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.fc1 = nn.Linear(n_head * d_v // 2, d_model // 2, bias=False)
            self.fc2 = nn.Linear(n_head * d_v // 2, d_model // 2, bias=False)

        self.dropout = nn.Dropout(residual_drop)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        if not self.partition:
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        else:
            q_c, q_p = q.chunk(2, dim=-1)
            k_c, k_p = k.chunk(2, dim=-1)
            v_c, v_p = v.chunk(2, dim=-1)

            q_c, q_p = self.w_qs_c(q_c).view(sz_b, len_q, n_head, d_k//2), self.w_qs_p(q_p).view(sz_b, len_q, n_head, d_k//2)
            k_c, k_p = self.w_ks_c(k_c).view(sz_b, len_k, n_head, d_k//2), self.w_ks_p(k_p).view(sz_b, len_k, n_head, d_k//2)
            v_c, v_p = self.w_vs_c(v_c).view(sz_b, len_v, n_head, d_v//2), self.w_vs_p(v_p).view(sz_b, len_v, n_head, d_v//2)

            q = torch.cat((q_c, q_p), dim=-1)
            k = torch.cat((k_c, k_p), dim=-1)
            v = torch.cat((v_c, v_p), dim=-1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        mask = mask.repeat(n_head, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        if not self.partition:
            output = (
                output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
            )  # b x lq x (n*dv)
            output = self.dropout(self.fc(output))
        else:
            output_c, output_p = output.chunk(2, dim=-1)
            output_c = output_c.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
            output_p = output_p.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
            output_c, output_p = self.dropout(self.fc1(output_c)), self.dropout(self.fc2(output_p))
            output = torch.cat((output_c, output_p), dim=-1)

        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, relu_drop=0.1, residual_drop=0.1, partition=False):
        super(PositionwiseFeedForward, self).__init__()
        self.partition = partition
        if not partition:
            self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
            self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        else:
            self.w_1c = nn.Linear(d_in//2, d_hid//2)  # position-wise
            self.w_2c = nn.Linear(d_hid//2, d_in//2)  # position-wise            
            self.w_1p = nn.Linear(d_in//2, d_hid//2)  # position-wise
            self.w_2p = nn.Linear(d_hid//2, d_in//2)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.relu_dropout = nn.Dropout(relu_drop)
        self.residual_dropout = nn.Dropout(residual_drop)

    def forward(self, x):
        residual = x
        if not self.partition:
            output = self.relu_dropout(F.relu(self.w_1(x)))
            output = self.residual_dropout(self.w_2(output))
        else:
            x_c, x_p = x.chunk(2, dim=-1)
            output_c = self.relu_dropout(F.relu(self.w_1c(x_c)))
            output_c = self.residual_dropout(self.w_2c(output_c))
            output_p = self.relu_dropout(F.relu(self.w_1p(x_p)))
            output_p = self.residual_dropout(self.w_2p(output_p))
            output = torch.cat((output_c, output_p), dim=-1)

        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, relu_drop=0.1, attention_drop=0.1, residual_drop=0.1, partition=False):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attention_drop=attention_drop, residual_drop=residual_drop, partition=partition)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, relu_drop=relu_drop, residual_drop=residual_drop, partition=partition)

    def forward(self, enc_input, mask):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=mask
        )
        pad_mask = (1 - mask).unsqueeze(-1)
        enc_output.masked_fill_(pad_mask, 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output.masked_fill_(pad_mask, 0)

        return enc_output, enc_slf_attn
