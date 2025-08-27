import React, { memo } from 'react';
import { AlertCircle, Info } from 'lucide-react';

interface FormInputProps {
  label: string;
  value: number | string;
  onChange: (value: number | string) => void;
  type?: 'number' | 'text';
  min?: number;
  max?: number;
  step?: number | string;
  unit?: string;
  placeholder?: string;
  error?: string;
  hint?: string;
  required?: boolean;
  disabled?: boolean;
  className?: string;
}

const FormInput = memo(function FormInput({
  label,
  value,
  onChange,
  type = 'number',
  min,
  max,
  step,
  unit,
  placeholder,
  error,
  hint,
  required = false,
  disabled = false,
  className = ''
}: FormInputProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = type === 'number' ? Number(e.target.value) : e.target.value;
    onChange(newValue);
  };

  const inputId = `input-${label.replace(/\s+/g, '-').toLowerCase()}`;

  return (
    <div className={className}>
      <label htmlFor={inputId} className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
        {unit && <span className="text-gray-500 ml-1">({unit})</span>}
      </label>
      
      <div className="relative">
        <input
          id={inputId}
          type={type}
          value={value}
          onChange={handleChange}
          min={min}
          max={max}
          step={step}
          placeholder={placeholder}
          disabled={disabled}
          className={`w-full px-3 py-2 border rounded-lg transition-colors focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed ${
            error 
              ? 'border-red-300 bg-red-50 focus:ring-red-500' 
              : 'border-gray-300 hover:border-gray-400'
          }`}
        />
        {error && (
          <AlertCircle className="absolute right-3 top-2.5 w-4 h-4 text-red-500" />
        )}
      </div>
      
      {error && (
        <p className="text-xs text-red-600 mt-1 flex items-center gap-1">
          <AlertCircle className="w-3 h-3 flex-shrink-0" />
          {error}
        </p>
      )}
      
      {hint && !error && (
        <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
          <Info className="w-3 h-3 flex-shrink-0" />
          {hint}
        </p>
      )}
    </div>
  );
});

// 材料选择组件
interface MaterialSelectorProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; description?: string }[];
  className?: string;
}

export const MaterialSelector = memo(function MaterialSelector({ value, onChange, options, className = '' }: MaterialSelectorProps) {
  return (
    <div className={className}>
      <label className="block text-sm font-medium text-gray-700 mb-2">材料类型</label>
      <div className="grid grid-cols-2 gap-2">
        {options.map((option) => (
          <button
            key={option.value}
            onClick={() => onChange(option.value)}
            className={`px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 border-2 ${
              value === option.value
                ? 'bg-blue-600 text-white border-blue-600 shadow-md transform scale-105'
                : 'bg-white text-gray-700 border-gray-200 hover:border-blue-300 hover:bg-blue-50'
            }`}
          >
            <div className="text-center">
              <div className="font-semibold">{option.label}</div>
              {option.description && (
                <div className="text-xs opacity-75 mt-1">{option.description}</div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
});

// 模式选择组件
interface ModeSelectorProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; icon?: React.ReactNode }[];
  className?: string;
}

export const ModeSelector = memo(function ModeSelector({ value, onChange, options, className = '' }: ModeSelectorProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      <label className="block text-sm font-medium text-gray-700">
        模式选择
      </label>
      <div className="flex gap-2">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
              value === option.value
                ? 'bg-blue-50 border-blue-200 text-blue-700'
                : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            {option.icon}
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
});

export default FormInput;