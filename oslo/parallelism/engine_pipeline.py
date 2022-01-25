from oslo.parallelism.mpu import MPU
from transformers import GPT2LMHeadModel
from transformers.utils.fx import symbolic_trace


class PipelineParallelEngine(object):
    def __init__(self, model, mpu):
        self.model = model
        self.mpu = mpu

    def parallelize(self):
        traced = symbolic_trace(self.model)
        for node in traced.graph.nodes:
            pass

    def request(self, node1, node2):
        pass

    def response(self, node1, node2):
        pass

    def cost_function(self, alpha):
        pass


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    mpu = MPU(tensor_parallel_size=1, pipeline_parallel_size=4)
    engine = PipelineParallelEngine(model, mpu)
    engine.parallelize()
