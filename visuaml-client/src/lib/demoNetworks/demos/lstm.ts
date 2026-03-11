/**
 * LSTM Demo - Recurrent neural network
 * Input → LSTM → Linear → Output
 */
import type { DemoNetwork } from '../types';

export const lstmDemo: DemoNetwork = {
  id: 'demo-lstm',
  name: 'LSTM Network',
  description: 'Long Short-Term Memory network for sequence processing',
  nodes: [
    {
      id: 'input',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'placeholder',
        name: 'x',
        label: 'Input',
        layerType: 'Input',
        outputShape: '[1, 10, 10]',
        color: '#4299e1',
        args: [],
        kwargs: {},
      },
    },
    {
      id: 'lstm',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'lstm',
        target: 'lstm',
        label: 'LSTM',
        layerType: 'LSTM',
        outputShape: '[1, 10, 20]',
        color: '#48bb78',
        args: [],
        kwargs: {
          input_size: 10,
          hidden_size: 20,
          num_layers: 1,
          batch_first: true,
        },
      },
    },
    {
      id: 'select',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_function',
        name: 'select',
        target: 'torch.select',
        label: 'Select Last',
        layerType: 'Select',
        outputShape: '[1, 20]',
        color: '#9f7aea',
        args: [],
        kwargs: {},
      },
    },
    {
      id: 'fc',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'fc',
        target: 'fc',
        label: 'Linear',
        layerType: 'Linear',
        outputShape: '[1, 5]',
        color: '#4299e1',
        args: [],
        kwargs: {
          in_features: 20,
          out_features: 5,
        },
      },
    },
  ],
  edges: [
    { id: 'e1', source: 'input', target: 'lstm' },
    { id: 'e2', source: 'lstm', target: 'select' },
    { id: 'e3', source: 'select', target: 'fc' },
  ],
};
