'use client'

import { useState, useEffect, useCallback } from 'react'
import { Document } from '@/types'
import { fetchDocuments } from '@/lib/api'

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [total,     setTotal]     = useState(0)
  const [search,    setSearch]    = useState('')
  const [loading,   setLoading]   = useState(true)

  // Initial load — top 20 by chunk count
  useEffect(() => {
    setLoading(true)
    fetchDocuments('', 20).then(({ documents, total }) => {
      setDocuments(documents)
      setTotal(total)
      setLoading(false)
    })
  }, [])

  // Search — re-fetch when search changes (debounced 300ms)
  useEffect(() => {
    if (!search) {
      // Reset to default top 20
      fetchDocuments('', 20).then(({ documents, total }) => {
        setDocuments(documents)
        setTotal(total)
      })
      return
    }
    const id = setTimeout(() => {
      fetchDocuments(search, 20).then(({ documents, total }) => {
        setDocuments(documents)
        setTotal(total)
      })
    }, 300)
    return () => clearTimeout(id)
  }, [search])

  return { documents, total, search, setSearch, loading }
}