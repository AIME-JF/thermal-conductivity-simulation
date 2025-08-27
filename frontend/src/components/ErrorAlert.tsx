import React from 'react';
import { AlertCircle, X, RefreshCw } from 'lucide-react';

interface ErrorAlertProps {
  error: string | null;
  onDismiss?: () => void;
  onRetry?: () => void;
  className?: string;
  variant?: 'error' | 'warning' | 'info';
}

const variantStyles = {
  error: {
    container: 'bg-red-50 border-red-200 text-red-800',
    icon: 'text-red-500',
    button: 'text-red-600 hover:text-red-800',
  },
  warning: {
    container: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    icon: 'text-yellow-500',
    button: 'text-yellow-600 hover:text-yellow-800',
  },
  info: {
    container: 'bg-blue-50 border-blue-200 text-blue-800',
    icon: 'text-blue-500',
    button: 'text-blue-600 hover:text-blue-800',
  },
};

export default function ErrorAlert({
  error,
  onDismiss,
  onRetry,
  className = '',
  variant = 'error'
}: ErrorAlertProps) {
  if (!error) return null;

  const styles = variantStyles[variant];

  return (
    <div className={`p-4 border rounded-lg ${styles.container} ${className}`}>
      <div className="flex items-start">
        <AlertCircle className={`w-5 h-5 ${styles.icon} mt-0.5 flex-shrink-0`} />
        <div className="ml-3 flex-1">
          <p className="text-sm font-medium">
            {variant === 'error' ? '操作失败' : variant === 'warning' ? '注意' : '提示'}
          </p>
          <p className="text-sm mt-1">{error}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className={`mt-2 inline-flex items-center gap-1 text-sm font-medium ${styles.button} hover:underline`}
            >
              <RefreshCw className="w-4 h-4" />
              重试
            </button>
          )}
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className={`ml-3 ${styles.button} hover:bg-opacity-20 hover:bg-current rounded p-1`}
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

// 简化的错误提示组件
export function SimpleError({ message, className = '' }: { message: string; className?: string }) {
  return (
    <div className={`flex items-center gap-2 text-red-600 text-sm ${className}`}>
      <AlertCircle className="w-4 h-4 flex-shrink-0" />
      <span>{message}</span>
    </div>
  );
}

// 成功提示组件
export function SuccessAlert({ message, onDismiss, className = '' }: { 
  message: string; 
  onDismiss?: () => void;
  className?: string;
}) {
  return (
    <div className={`p-4 bg-green-50 border border-green-200 rounded-lg ${className}`}>
      <div className="flex items-start">
        <div className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0">
          <svg fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        </div>
        <div className="ml-3 flex-1">
          <p className="text-sm font-medium text-green-800">操作成功</p>
          <p className="text-sm text-green-700 mt-1">{message}</p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="ml-3 text-green-600 hover:text-green-800 hover:bg-green-100 rounded p-1"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}