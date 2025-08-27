import React, { useState, useRef, memo } from 'react';
import { Upload, Download, FileText, AlertCircle, CheckCircle } from 'lucide-react';

interface BatchResult {
  total_rows: number;
  success_count: number;
  error_count: number;
  results: Array<{
    row_index: number;
    success: boolean;
    result?: any;
    error?: string;
  }>;
  processing_time: number;
}

interface BatchProcessorProps {
  method: 'steady' | 'quasi' | 'cooling';
  material: string;
  onResultsChange: (results: BatchResult | null) => void;
}

const BatchProcessor: React.FC<BatchProcessorProps> = ({ 
  method, 
  material, 
  onResultsChange 
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.csv')) {
        setError('请选择CSV文件');
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResults(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('请先选择文件');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', method);
      if (material) {
        formData.append('material', material);
      }

      const response = await fetch('/api/batch/upload-csv', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '处理失败');
      }

      const result: BatchResult = await response.json();
      setResults(result);
      onResultsChange?.(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '处理失败';
      setError(errorMessage);
      setResults(null);
      onResultsChange?.(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;

    const csvContent = generateResultsCSV(results);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `batch_results_${method}_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const generateResultsCSV = (results: BatchResult): string => {
    const headers = ['行号', '状态', '错误信息'];
    
    // 根据方法添加结果列
    if (method === 'steady') {
      headers.push('导热系数', '修正后T2', '温差变化率');
    } else if (method === 'quasi') {
      headers.push('导热系数', '热扩散率', '比热容');
    } else if (method === 'cooling') {
      headers.push('最高温度', '最低温度', '平均冷却速率', '半冷却时间');
    }

    const rows = [headers.join(',')];
    
    results.results.forEach(item => {
      const row = [
        item.row_index.toString(),
        item.success ? '成功' : '失败',
        item.error || ''
      ];
      
      if (item.success && item.result) {
        if (method === 'steady') {
          row.push(
            item.result.thermal_conductivity?.toString() || '',
            item.result.corrected_T2?.toString() || '',
            item.result.temperature_difference_rate?.toString() || ''
          );
        } else if (method === 'quasi') {
          row.push(
            item.result.thermal_conductivity?.toString() || '',
            item.result.thermal_diffusivity?.toString() || '',
            item.result.specific_heat?.toString() || ''
          );
        } else if (method === 'cooling') {
          row.push(
            item.result.max_temp?.toString() || '',
            item.result.min_temp?.toString() || '',
            item.result.avg_cooling_rate?.toString() || '',
            item.result.time_to_half_temp?.toString() || ''
          );
        }
      } else {
        // 添加空值以保持列对齐
        if (method === 'steady') {
          row.push('', '', '');
        } else if (method === 'quasi') {
          row.push('', '', '');
        } else if (method === 'cooling') {
          row.push('', '', '', '');
        }
      }
      
      rows.push(row.join(','));
    });
    
    return rows.join('\n');
  };

  const downloadTemplate = () => {
    let csvContent = '';
    
    if (method === 'steady') {
      csvContent = 'T1,T2\n100,50\n120,60\n90,40';
    } else if (method === 'quasi') {
      csvContent = 'V_t,delta_V,material\n5.2,0.1,glass\n4.8,0.15,rubber\n6.0,0.12,glass';
    } else if (method === 'cooling') {
      csvContent = 'duration,noise\n300,0.1\n600,0.2\n450,0.15';
    }
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `template_${method}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">批量处理</h3>
        <button
          onClick={downloadTemplate}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
        >
          <Download className="w-4 h-4" />
          下载模板
        </button>
      </div>

      {/* 文件上传区域 */}
      <div className="mb-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          {file ? (
            <div className="flex items-center justify-center gap-2 text-green-600">
              <FileText className="w-5 h-5" />
              <span>{file.name}</span>
            </div>
          ) : (
            <div className="text-gray-500">
              <Upload className="w-8 h-8 mx-auto mb-2" />
              <p>点击选择CSV文件或拖拽文件到此处</p>
              <p className="text-sm mt-1">支持格式: .csv</p>
            </div>
          )}
          
          <button
            onClick={() => fileInputRef.current?.click()}
            className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            选择文件
          </button>
        </div>
      </div>

      {/* 处理按钮 */}
      <div className="mb-6">
        <button
          onClick={handleUpload}
          disabled={!file || isProcessing}
          className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {isProcessing ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              处理中...
            </>
          ) : (
            '开始批量处理'
          )}
        </button>
      </div>

      {/* 错误信息 */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}

      {/* 处理结果 */}
      {results && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-blue-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{results.total_rows}</div>
              <div className="text-sm text-blue-600">总行数</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-600">{results.success_count}</div>
              <div className="text-sm text-green-600">成功</div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-red-600">{results.error_count}</div>
              <div className="text-sm text-red-600">失败</div>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">
              处理时间: {results.processing_time.toFixed(2)}秒
            </span>
            <button
              onClick={downloadResults}
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Download className="w-4 h-4" />
              下载结果
            </button>
          </div>

          {/* 错误详情 */}
          {results.error_count > 0 && (
            <div className="max-h-40 overflow-y-auto">
              <h4 className="text-sm font-medium text-gray-700 mb-2">错误详情:</h4>
              <div className="space-y-1">
                {results.results
                  .filter(item => !item.success)
                  .map(item => (
                    <div key={item.row_index} className="text-sm text-red-600 bg-red-50 p-2 rounded">
                      第{item.row_index}行: {item.error}
                    </div>
                  ))
                }
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default memo(BatchProcessor);