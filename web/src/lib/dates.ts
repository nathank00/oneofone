/**
 * Get today's date string in US/Eastern timezone as "YYYY-MM-DD".
 * Uses 'en-CA' locale which outputs ISO format (YYYY-MM-DD).
 */
export function getTodayEastern(): string {
  return new Date().toLocaleDateString("en-CA", {
    timeZone: "America/New_York",
  });
}

/**
 * Convert a "YYYY-MM-DD" date string to the GAME_DATE format stored in
 * Supabase: "YYYY-MM-DDT00:00:00+00:00".
 *
 * GAME_DATE is always the EST game date at midnight — no real UTC
 * conversion is needed because the pipeline normalizes all dates to
 * this format.
 */
export function toGameDate(dateStr: string): string {
  return `${dateStr}T00:00:00+00:00`;
}
