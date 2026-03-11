/**
 * Simple MLP Demo
 * Architecture: Linear → ReLU → Linear → ReLU → Linear
 * Based on: models.SimpleNN
 */
import type { DemoNetwork } from '../types';

export const simpleMLPDemo: DemoNetwork = {
  id: 'demo-simple-mlp',
  name: 'Simple MLP',
  description: 'A multi-layer perceptron for classification',
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
        outputShape: '[1, 784]',
        color: '#4299e1',
        args: [],
        kwargs: {},
      },
    },
    {
      id: 'fc1',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'fc1',
        target: 'fc1',
        label: 'Linear',
        layerType: 'Linear',
        outputShape: '[1, 512]',
        color: '#4299e1',
        args: [],
        kwargs: {
          in_features: 784,
          out_features: 512,
        },
      },
    },
    {
      id: 'relu1',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'relu1',
        target: 'relu1',
        label: 'ReLU',
        layerType: 'ReLU',
        outputShape: '[1, 512]',
        color: '#ed8936',
        args: [],
        kwargs: {},
      },
    },
    {
      id: 'fc2',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'fc2',
        target: 'fc2',
        label: 'Linear',
        layerType: 'Linear',
        outputShape: '[1, 256]',
        color: '#4299e1',
        args: [],
        kwargs: {
          in_features: 512,
          out_features: 256,
        },
      },
    },
    {
      id: 'relu2',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'relu2',
        target: 'relu2',
        label: 'ReLU',
        layerType: 'ReLU',
        outputShape: '[1, 256]',
        color: '#ed8936',
        args: [],
        kwargs: {},
      },
    },
    {
      id: 'fc3',
      type: 'transformer',
      position: { x: 0, y: 0 },
      data: {
        op: 'call_module',
        name: 'fc3',
        target: 'fc3',
        label: 'Linear',
        layerType: 'Linear',
        outputShape: '[1, 10]',
        color: '#4299e1',
        args: [],
        kwargs: {
          in_features: 256,
          out_features: 10,
        },
      },
    },
  ],
  edges: [
    { id: 'e1', source: 'input', target: 'fc1' },
    { id: 'e2', source: 'fc1', target: 'relu1' },
    { id: 'e3', source: 'relu1', target: 'fc2' },
    { id: 'e4', source: 'fc2', target: 'relu2' },
    { id: 'e5', source: 'relu2', target: 'fc3' },
  ],
};
