'use client'

import { useState, useEffect } from 'react'
import { SystemStats } from '@/types'
import { fetchStats } from '@/lib/api'

export function useStats() {
  const [stats, setStats] = useState<SystemStats | null>(null)

  useEffect(() => {
    // Fetch immediately on mount
    fetchStats().then(s => { if (s) setStats(s) })

    // Poll every 30 seconds
    const id = setInterval(() => {
      fetchStats().then(s => { if (s) setStats(s) })
    }, 30_000)

    return () => clearInterval(id)
  }, [])

  return stats
}