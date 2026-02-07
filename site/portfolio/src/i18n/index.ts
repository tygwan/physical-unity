import { en, type Translations } from './en';
import { ko } from './ko';

export type Lang = 'en' | 'ko';

const translations: Record<Lang, Translations> = { en, ko };

export function getTranslations(lang: Lang): Translations {
  return translations[lang] ?? en;
}

export function getOtherLang(lang: Lang): Lang {
  return lang === 'en' ? 'ko' : 'en';
}

export function getLangLabel(lang: Lang): string {
  return lang === 'en' ? 'English' : '한국어';
}

export type { Translations };
