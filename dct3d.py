import torch
import torch.nn as nn
import math
from dct3d_function import DCT3DFunction


class DCT3D(nn.Module):
    def __init__(self, freqs):
         super(DCT3D, self).__init__()
         self.freqs = freqs
 
    def forward(self, input):
        B, C, M, N = input.shape
        A_col = torch.cos(torch.mm(torch.arange(N).float().unsqueeze(0).t(), (torch.arange(N).float()+0.5).unsqueeze(0))*math.pi/N)*torch.sqrt(torch.tensor(2.0)/N)
        A_col[0, :] = A_col[0, :] / torch.sqrt(torch.tensor(2.0))
        A_col = A_col.type(torch.cuda.FloatTensor)
        A_row = torch.cos(torch.mm(torch.arange(M).float().unsqueeze(0).t(), (torch.arange(M).float()+0.5).unsqueeze(0))*math.pi/M)*torch.sqrt(torch.tensor(2.0)/M)
        A_row[0, :] = A_row[0, :] / torch.sqrt(torch.tensor(2.0))
        A_row = A_row.type(torch.cuda.FloatTensor)
        A_cnl = torch.cos(torch.mm(torch.arange(C).float().unsqueeze(0).t(), (torch.arange(C).float()+0.5).unsqueeze(0))*math.pi/C)*torch.sqrt(torch.tensor(2.0)/C)
        A_cnl[0, :] = A_cnl[0, :] / torch.sqrt(torch.tensor(2.0))
        A_cnl = A_cnl.type(torch.cuda.FloatTensor)

        col = input.view(B*C*M, N).transpose(0, 1)
        D_col = A_col.mm(col)
        row = D_col.view(N, B*C, M).transpose(0, 2).contiguous().view(M, B*C*N)
        D_row = A_row.mm(row)
        cnl = D_row.view(M, B, C, N).transpose(0, 2).contiguous().view(C, B*M*N)
        D_cnl = A_cnl.mm(cnl)

        size = D_cnl.shape[0]*D_cnl.shape[1]
        x = D_cnl.view(size)
        freqs = [B*C*M*N/4, B*C*M*N/16, B*C*M*N/64, B*C*M*N/256]
        for i in freqs:
            mark = abs(x).topk(i)[0][-1]
            mask = abs(x).ge(mark)
            temp = mask.type(torch.cuda.FloatTensor)*x
            D_cnl = temp.view(D_cnl.shape)

            ID_cnl = A_cnl.t().mm(D_cnl)  # ID_cnl == cnl
            ID_cnl_ = ID_cnl.view(C, B, M, N).transpose(0, 2).contiguous().view(M, B*C*N)  # temp == D_row
            ID_row = A_row.t().mm(ID_cnl_)  # ID_row == row
            ID_row_ = ID_row.view(M, B*C, N).transpose(0, 2).contiguous().view(N, B*C*M)
            ID_col = A_col.t().mm(ID_row_)  # ID_col == col
            ID_col_ = ID_col.transpose(0, 1).view(B, C, M, N)
            if i == freqs[0]:
                output = ID_col_.unsqueeze(0)
            else:
                output = torch.cat((output, ID_col_.unsqueeze(0)), 0)

        return output
        