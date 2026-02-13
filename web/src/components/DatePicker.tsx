"use client";

interface DatePickerProps {
  value: string;
  onChange: (date: string) => void;
}

export default function DatePicker({ value, onChange }: DatePickerProps) {
  return (
    <input
      type="date"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2
                 text-sm text-neutral-100 outline-none transition-colors
                 focus:border-neutral-500 cursor-pointer
                 [color-scheme:dark]"
    />
  );
}
