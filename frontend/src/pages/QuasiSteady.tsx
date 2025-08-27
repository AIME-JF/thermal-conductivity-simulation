import React, { useState, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Calculator, Upload, Download, FileText, Settings, Thermometer, Zap } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { quasiSteadyAPI, QuasiSteadyRequest, QuasiSteadyResponse } from '@/lib/api';
import LoadingSpinner, { ButtonLoader } from '../components/LoadingSpinner';
import ErrorAlert, { SuccessAlert } from '../components/ErrorAlert';
import FormInput, { MaterialSelector } from '../components/FormInput';
import BatchProcessor from '../components/BatchProcessor';
import ModelSelector from '../components/ModelSelector';

interface FormData {
  V_t: number;
  delta_V: number;
  material: 'glass' | 'rubber';
  selectedModel: string; // 添加模型选择字段
  // 材料参数（可编辑）
  U: number;
  resistance: number;
  area: number;
  thickness: number;
  density: number;
}

interface HistoryRecord {
  id: string;
  timestamp: string;
  input: FormData;
  result: QuasiSteadyResponse;
}

const MATERIAL_DEFAULTS = {
  glass: {
    U: 12.0,
    resistance: 50.0,
    area: 0.01,
    thickness: 0.005,
    density: 2500,
  },
  rubber: {
    U: 8.0,
    resistance: 75.0,
    area: 0.01,
    thickness: 0.008,
    density: 1200,
  },
};

export default function QuasiSteady() {
  const [formData, setFormData] = useState<FormData>({
    V_t: 0.014,
    delta_V: 0.022,
    material: 'glass',
    selectedModel: 'default', // 默认使用系统模型
    ...MATERIAL_DEFAULTS.glass,
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<QuasiSteadyResponse | null>(null);
  

  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showBatch, setShowBatch] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const validateInput = (field: keyof FormData, value: number): string | null => {
    switch (field) {
      case 'V_t':
        if (value < 0.001 || value > 1) return '电压范围应在 0.001-1 mV 之间';
        break;
      case 'delta_V':
        if (value < 0.001 || value > 1) return '电压变化率范围应在 0.001-1 mV/min 之间';
        break;
      case 'U':
        if (value < 1 || value > 50) return '电压U范围应在 1-50 V 之间';
        break;
      case 'resistance':
        if (value < 1 || value > 1000) return '电阻范围应在 1-1000 Ω 之间';
        break;
      case 'area':
        if (value < 0.0001 || value > 1) return '面积范围应在 0.0001-1 m² 之间';
        break;
      case 'thickness':
        if (value < 0.001 || value > 0.1) return '厚度范围应在 0.001-0.1 m 之间';
        break;
      case 'density':
        if (value < 100 || value > 10000) return '密度范围应在 100-10000 kg/m³ 之间';
        break;
    }
    return null;
  };

  const handleInputChange = (field: keyof FormData, value: number | string) => {
    // 清除之前的验证错误
    setValidationErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[field];
      return newErrors;
    });
    
    // 对于字符串字段（如selectedModel），直接使用原值
    if (field === 'selectedModel' || field === 'material') {
      setFormData(prev => ({ ...prev, [field]: value }));
      return;
    }
    
    // 对于数字字段，进行转换和验证
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    
    // 验证输入
    if (!isNaN(numValue)) {
      const validationError = validateInput(field, numValue);
      if (validationError) {
        setValidationErrors(prev => ({ ...prev, [field]: validationError }));
      }
    }
    
    setFormData(prev => ({ ...prev, [field]: numValue }));
  };

  const handleMaterialChange = (material: string) => {
    const materialKey = material as 'glass' | 'rubber';
    const defaults = MATERIAL_DEFAULTS[materialKey];
    if (defaults) {
      setFormData(prev => ({
        ...prev,
        material: materialKey,
        ...defaults
      }));
    }
  };

  const hasValidationErrors = Object.keys(validationErrors).length > 0;
  const isFormValid = !hasValidationErrors && 
    formData.V_t > 0 && formData.delta_V > 0 && 
    formData.U > 0 && formData.resistance > 0 && 
    formData.area > 0 && formData.thickness > 0 && formData.density > 0;

  const handlePredict = useCallback(async (retryCount = 0) => {
    if (!isFormValid) {
      setError('请检查并修正所有输入参数');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      
      const request: QuasiSteadyRequest = {
        V_t: formData.V_t,
        delta_V: formData.delta_V,
        material: formData.material,
        selectedModel: formData.selectedModel,
        constantsOverride: {
          U: formData.U,
          resistance: formData.resistance,
          area: formData.area,
          thickness: formData.thickness,
          density: formData.density,
        },
      };
      
      const result = await quasiSteadyAPI.predict(request);
        
        // 详细调试信息
        console.log('=== API调用调试信息 ===');
        console.log('请求参数:', request);
        console.log('API返回的原始结果:', result);
        console.log('lambda_theory值:', result.lambda_theory, '类型:', typeof result.lambda_theory);
        console.log('lambda_predicted值:', result.lambda_predicted, '类型:', typeof result.lambda_predicted);
        console.log('intermediate_values:', result.intermediate_values);
        console.log('intermediate_values中的lambda_theory:', result.intermediate_values?.lambda_theory);
        console.log('========================');
        
        setResults(result);
      
      // 添加到历史记录
      const record: HistoryRecord = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleString(),
        input: { ...formData },
        result,
      };
      setHistory(prev => [record, ...prev.slice(0, 9)]); // 保留最近10条记录
      
      setSuccess('预测完成！结果已更新。');
    } catch (err: any) {
      const errorMessage = err.message || err.response?.data?.detail || '预测失败';
      
      // 网络错误且重试次数小于3次时自动重试
      if (!err.response && retryCount < 3) {
        setTimeout(() => {
          handlePredict(retryCount + 1);
        }, 1000 * (retryCount + 1)); // 递增延迟重试
        setError(`网络连接失败，正在重试... (${retryCount + 1}/3)`);
      } else {
        setError(errorMessage);
      }
    } finally {
      if (retryCount === 0) {
        setLoading(false);
      }
    }
  }, [formData, isFormValid]);

  const exportData = () => {
    if (history.length === 0) return;
    
    const csvContent = [
      ['时间戳', '材料', 'V_t(mV)', 'ΔV(mV/min)', 'λ预测', 'λ理论', 'λ误差(%)'],
      ...history.map(record => [
        record.timestamp,
        record.input.material,
        record.input.V_t,
        record.input.delta_V,
        record.result.lambda_predicted.toFixed(4),
        record.result.lambda_theory.toFixed(4),
        record.result.lambda_error.toFixed(2),
      ]),
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `quasi_steady_results_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
  };

  // 准备对比图表数据（使用useMemo优化性能）
  const comparisonData = useMemo(() => {
    return results ? [
      {
        name: '导热系数 λ',
        predicted: results.lambda_predicted,
        theory: results.lambda_theory,
        error: Math.abs(results.lambda_error),
      },
    ] : [];
  }, [results]);

  // 历史记录表格行数据
  const historyTableRows = useMemo(() => 
    history.slice(0, 5).map((record) => (
      <tr key={record.id} className="hover:bg-gray-50">
        <td className="px-4 py-2 text-sm text-gray-900">
          {record.timestamp.split(' ')[1]}
        </td>
        <td className="px-4 py-2 text-sm text-gray-900">
          {record.input.material === 'glass' ? '有机玻璃' : '橡胶'}
        </td>
        <td className="px-4 py-2 text-sm text-gray-900">
          {record.input.V_t.toFixed(3)}
        </td>
        <td className="px-4 py-2 text-sm text-gray-900">
          {record.input.delta_V.toFixed(3)}
        </td>
        <td className="px-4 py-2 text-sm text-gray-900">
          {record.result.lambda_predicted.toFixed(4)}
        </td>
        <td className={`px-4 py-2 text-sm font-medium ${
          Math.abs(record.result.lambda_error) < 5 
            ? 'text-green-600' 
            : Math.abs(record.result.lambda_error) < 10 
            ? 'text-yellow-600' 
            : 'text-red-600'
        }`}>
          {record.result.lambda_error.toFixed(2)}%
        </td>
      </tr>
    )), [history]
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 头部导航 */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/" className="flex items-center gap-2 text-gray-600 hover:text-gray-900">
                <ArrowLeft className="w-5 h-5" />
                返回
              </Link>
              <h1 className="text-2xl font-bold text-gray-900">准稳态法实验台</h1>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={exportData}
                disabled={history.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Download className="w-4 h-4" />
                导出历史
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* 左侧参数面板 */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">实验参数</h2>
              
              {/* 模型选择 */}
              <ModelSelector
                value={formData.selectedModel}
                onChange={(value) => handleInputChange('selectedModel', value)}
                modelType="quasi"
                className="mb-6"
              />
              
              {/* 材料选择 */}
              <MaterialSelector
                value={formData.material}
                onChange={(value) => handleMaterialChange(value)}
                options={[
                  { value: 'glass', label: '有机玻璃', description: '透明固体材料' },
                  { value: 'rubber', label: '橡胶', description: '弹性聚合物' }
                ]}
                className="mb-6"
              />

              {/* 电压参数 */}
              <div className="space-y-4">
                <FormInput
                  label="电压 V_t"
                  value={formData.V_t}
                  onChange={(value) => handleInputChange('V_t', value as number)}
                  type="number"
                  unit="mV"
                  step={0.001}
                  min={0.001}
                  max={1}
                  error={validationErrors.V_t}
                  hint="范围: 0.001-1 mV"
                  required
                />
                
                <FormInput
                  label="电压变化率 ΔV"
                  value={formData.delta_V}
                  onChange={(value) => handleInputChange('delta_V', value as number)}
                  type="number"
                  unit="mV/min"
                  step={0.001}
                  min={0.001}
                  max={1}
                  error={validationErrors.delta_V}
                  hint="范围: 0.001-1 mV/min"
                  required
                />
              </div>

              {/* 高级设置 */}
              <div className="mt-6">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
                >
                  <Settings className="w-4 h-4" />
                  材料参数设置
                </button>
                
                {showAdvanced && (
                  <div className="mt-4 space-y-4 p-4 bg-gray-50 rounded-lg">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        电压 U (V)
                      </label>
                      <input
                        type="number"
                        value={formData.U}
                        onChange={(e) => handleInputChange('U', Number(e.target.value))}
                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent ${
                          validationErrors.U ? 'border-red-300 bg-red-50' : 'border-gray-300'
                        }`}
                        step="0.1"
                      />
                      {validationErrors.U && (
                        <p className="text-xs text-red-600 mt-1">{validationErrors.U}</p>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        电阻 (Ω)
                      </label>
                      <input
                        type="number"
                        value={formData.resistance}
                        onChange={(e) => handleInputChange('resistance', Number(e.target.value))}
                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent ${
                          validationErrors.resistance ? 'border-red-300 bg-red-50' : 'border-gray-300'
                        }`}
                        step="0.1"
                      />
                      {validationErrors.resistance && (
                        <p className="text-xs text-red-600 mt-1">{validationErrors.resistance}</p>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        面积 (m²)
                      </label>
                      <input
                        type="number"
                        value={formData.area}
                        onChange={(e) => handleInputChange('area', Number(e.target.value))}
                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent ${
                          validationErrors.area ? 'border-red-300 bg-red-50' : 'border-gray-300'
                        }`}
                        step="0.001"
                      />
                      {validationErrors.area && (
                        <p className="text-xs text-red-600 mt-1">{validationErrors.area}</p>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        厚度 (m)
                      </label>
                      <input
                        type="number"
                        value={formData.thickness}
                        onChange={(e) => handleInputChange('thickness', Number(e.target.value))}
                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent ${
                          validationErrors.thickness ? 'border-red-300 bg-red-50' : 'border-gray-300'
                        }`}
                        step="0.001"
                      />
                      {validationErrors.thickness && (
                        <p className="text-xs text-red-600 mt-1">{validationErrors.thickness}</p>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        密度 (kg/m³)
                      </label>
                      <input
                        type="number"
                        value={formData.density}
                        onChange={(e) => handleInputChange('density', Number(e.target.value))}
                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent ${
                          validationErrors.density ? 'border-red-300 bg-red-50' : 'border-gray-300'
                        }`}
                        step="10"
                      />
                      {validationErrors.density && (
                        <p className="text-xs text-red-600 mt-1">{validationErrors.density}</p>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* 操作按钮 */}
              <div className="mt-6 space-y-3">
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowBatch(false)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      !showBatch 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    单次实验
                  </button>
                  <button
                    onClick={() => setShowBatch(true)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      showBatch 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    批量处理
                  </button>
                </div>
                
                {!showBatch && (
                  <button
                     onClick={() => handlePredict()}
                     disabled={loading || !isFormValid}
                     className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                       loading || !isFormValid 
                         ? 'bg-gray-400 cursor-not-allowed' 
                         : 'bg-green-600 hover:bg-green-700'
                     } text-white`}
                   >
                    {loading ? (
                      <ButtonLoader text="预测中..." />
                    ) : (
                      <>
                        <Calculator className="w-4 h-4" />
                        {!isFormValid ? '请检查参数' : '开始预测'}
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* 状态提示 */}
              <ErrorAlert 
                error={error} 
                onDismiss={() => setError(null)}
                onRetry={handlePredict}
                className="mt-4"
              />
              
              {success && (
                <SuccessAlert 
                  message={success}
                  onDismiss={() => setSuccess(null)}
                  className="mt-4"
                />
              )}
            </div>
          </div>

          {/* 右侧结果和图表 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 批量处理组件 */}
            {showBatch && (
              <BatchProcessor
                method="quasi"
                material={formData.material}
                onResultsChange={(results) => {
                  if (results) {
                    console.log('批量处理完成:', results);
                    setSuccess(`批量处理完成！成功处理 ${results.success_count}/${results.total_rows} 个样本`);
                  }
                }}
              />
            )}
            
            {/* 预测结果 */}
            {!showBatch && results && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">预测结果</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">导热系数 λ</h4>
                      <div className="space-y-1">
                        <p className="text-xl font-bold text-blue-600">
                          {results.lambda_predicted.toFixed(4)} W/(m·K)
                        </p>
                        <p className="text-sm text-blue-700">
                          理论值: {results.lambda_theory.toFixed(4)} W/(m·K)
                        </p>
                        <p className="text-sm text-blue-700">
                          误差: {results.lambda_error.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                    

                  </div>
                  
                  <div className="space-y-4">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">模型信息</h4>
                      <p className="text-sm text-gray-600">
                        版本: {results.model_version}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">中间计算值</h4>
                      <div className="space-y-1 text-sm">
                        {Object.entries(results.intermediate_values).map(([key, value]) => (
                          <p key={key}>
                            {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                          </p>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 预测 vs 理论对比图 */}
            {results && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">预测值 vs 理论值对比</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value: number, name: string) => [
                          value.toFixed(4),
                          name === 'predicted' ? '预测值' : '理论值'
                        ]}
                      />
                      <Legend />
                      <Bar dataKey="predicted" fill="#10b981" name="预测值" />
                      <Bar dataKey="theory" fill="#6b7280" name="理论值" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* 历史记录 */}
            {!showBatch && history.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">历史记录</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">时间</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">材料</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">V_t</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">ΔV</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">λ预测</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">λ误差</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {historyTableRows}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* 实验说明 */}
            {!showBatch && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">实验说明</h3>
                <div className="prose text-sm text-gray-600">
                  <p className="mb-3">
                    <strong>准稳态法实验</strong>通过测量材料在准稳态传热条件下的电势变化，同时预测材料的导热系数和比热容。
                  </p>
                  <p className="mb-3">
                    <strong>物理原理：</strong>基于非稳态传热理论，利用电势信号的时间变化特征，通过多任务神经网络同时预测材料的热物性参数。
                  </p>
                  <p className="mb-3">
                    <strong>应用场景：</strong>复合材料热物性测试、新材料研发、热管理系统设计、材料质量控制等。
                  </p>
                  <p>
                    <strong>参数说明：</strong>
                  </p>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>V_t: 电势值，反映材料内部温度分布的电信号</li>
                    <li>ΔV: 电势变化率，表示电势随时间的变化速度</li>
                    <li>U: 电压，实验中施加的电压值</li>
                    <li>R: 电阻，材料的电阻特性</li>
                    <li>A: 面积，材料的有效传热面积</li>
                    <li>d: 厚度，材料的厚度尺寸</li>
                    <li>ρ: 密度，材料的密度参数</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}