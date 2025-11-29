import * as React from 'react'

interface CuboLogoProps {
  className?: string
  size?: number
}

export function CuboLogo({ className, size = 24 }: CuboLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* 3D Cube */}
      <path
        d="M12 2L2 7L12 12L22 7L12 2Z"
        className="fill-primary"
      />
      <path
        d="M2 7V17L12 22V12L2 7Z"
        className="fill-primary/80"
      />
      <path
        d="M22 7V17L12 22V12L22 7Z"
        className="fill-primary/60"
      />
    </svg>
  )
}
