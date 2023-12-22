from transformers import PretrainedConfig


class TrajGLMConfig(PretrainedConfig):
    model_type = "TrajGLM"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        interleaved_qkv=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=None,
        pre_seq_len=None,
        prefix_projection=False,
        road_hidden_size=128,
        time_hidden_size=128,
        has_road_source=True,
        has_time_source=False,
        road_source_path="",
        total_minute_size=1501,
        total_week_size=8,
        add_bias_st_linear=False,
        st_hidden_size = 128,
        total_st_size = 40307,
        total_time_size = 1509,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.road_hidden_size = road_hidden_size
        self.time_hidden_size = time_hidden_size
        self.has_road_source=has_road_source
        self.has_time_source=has_time_source
        self.road_source_path=road_source_path
        self.total_minute_size = total_minute_size
        self.total_week_size = total_week_size
        self.add_bias_st_linear = add_bias_st_linear
        self.st_hidden_size = st_hidden_size
        self.total_st_size = total_st_size
        self.total_time_size = total_time_size
        super().__init__(**kwargs)