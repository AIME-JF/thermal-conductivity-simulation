import React, { useState, useEffect, memo } from 'react';
import { Settings, Database, RefreshCw, AlertCircle } from 'lucide-react';
import { adminAPI, ModelInfo } from '@/lib/api';

interface ModelSelectorProps {
  value: string;
  onChange: (modelPath: string) => void;
  modelType: 'steady' | 'quasi';
  className?: string;
}

interface ModelOption {
  value: string;
  label: string;
  description: string;
  isDefault?: boolean;
}

const ModelSelector = memo(function ModelSelector({ value, onChange, modelType, className = '' }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const modelList = await adminAPI.getModels();
      setModels(modelList);
    } catch (err: any) {
      setError('获取模型列表失败');
      console.error('Failed to fetch models:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const getRelevantModels = (): ModelOption[] => {
    const options: ModelOption[] = [
      {
        value: 'default',
        label: '默认模型',
        description: '系统内置的预训练模型',
        isDefault: true
      }
    ];

    const relevantModels = models.filter(model => {
      const modelName = model.name.toLowerCase();
      if (modelType === 'steady') {
        return modelName.includes('steady') || modelName.includes('稳态');
      } else {
        return modelName.includes('quasi') || modelName.includes('准稳态');
      }
    });

    relevantModels.forEach(model => {
      options.push({
        value: model.file_path,
        label: model.name,
        description: `训练于 ${new Date(model.timestamp).toLocaleDateString()}, 版本: ${model.version}, 大小: ${model.size_mb}MB`
      });
    });

    return options;
  };

  const modelOptions = getRelevantModels();
  const selectedOption = modelOptions.find(option => option.value === value) || modelOptions[0];

  return (
    <div className={`relative ${className}`}>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        模型选择
      </label>
      
      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-left shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            {selectedOption?.isDefault ? (
              <Settings className="w-4 h-4 text-blue-500" />
            ) : (
              <Database className="w-4 h-4 text-green-500" />
            )}
            <div>
              <div className="text-sm font-medium text-gray-900">
                {selectedOption?.label || '选择模型'}
              </div>
              {selectedOption?.description && (
                <div className="text-xs text-gray-500">
                  {selectedOption.description}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            {loading && <RefreshCw className="w-4 h-4 animate-spin text-gray-400" />}
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </button>

        {isOpen && (
          <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto">
            {error ? (
              <div className="px-3 py-2 text-sm text-red-600 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            ) : (
              <div className="py-1">
                {modelOptions.map((option) => (
                  <div
                    key={option.value}
                    onClick={() => {
                      onChange(option.value);
                      setIsOpen(false);
                    }}
                    className={`w-full px-3 py-2 text-left hover:bg-gray-50 flex items-center gap-2 cursor-pointer ${
                      option.value === value ? 'bg-blue-50 text-blue-700' : 'text-gray-900'
                    }`}
                  >
                    {option.isDefault ? (
                      <Settings className="w-4 h-4 text-blue-500" />
                    ) : (
                      <Database className="w-4 h-4 text-green-500" />
                    )}
                    <div className="flex-1">
                      <div className="text-sm font-medium">{option.label}</div>
                      <div className="text-xs text-gray-500">{option.description}</div>
                    </div>
                    {option.value === value && (
                      <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {isOpen && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setIsOpen(false)}
        ></div>
      )}
    </div>
  );
});

export default ModelSelector;