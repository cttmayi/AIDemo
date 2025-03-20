import torch
import torch.nn as nn

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, num_heads, num_layers, dim_feedforward, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        """
        tgt: 目标序列 (batch_size, tgt_seq_length)
        memory: 编码器的输出 (batch_size, src_seq_length, d_model)
        """
        tgt_embeddings = self.embedding(tgt) + self.positional_encoding[:tgt.shape[1], :]
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        output = self.transformer_decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)
        return self.fc(output)

    def generate_square_subsequent_mask(self, sz):
        """
        生成掩码矩阵，掩盖未来信息。
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, num_heads, num_layers, dim_feedforward, max_seq_length):
        super().__init__()
        self.encoder = TransformerEncoderModel(vocab_size, embedding_dim, d_model, num_heads, num_layers, dim_feedforward, max_seq_length)
        self.decoder = TransformerDecoderModel(vocab_size, embedding_dim, d_model, num_heads, num_layers, dim_feedforward, max_seq_length)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        return decoder_output

# 示例数据
src_data = torch.randint(0, 10, (5, 10))  # 源序列
tgt_data = torch.randint(0, 10, (5, 10))  # 目标序列

# 模型
model = TransformerModel(vocab_size, embedding_dim, d_model, num_heads, num_layers, dim_feedforward, max_seq_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(src_data, tgt_data)
    loss = criterion(outputs.view(-1, vocab_size), tgt_data.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")