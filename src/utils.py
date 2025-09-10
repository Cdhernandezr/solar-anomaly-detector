class Logger:
    """Sistema de logging personalizado para el proyecto"""
    STYLES = {
        'header': '\033[95m\033[1m',
        'info': '\033[94m',
        'success': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }

    @classmethod
    def log(cls, message, style='info'):
        print(f"{cls.STYLES[style]}{message}{cls.STYLES['end']}")

    @classmethod
    def section(cls, title):
        print(f"\n{cls.STYLES['header']}{'='*80}")
        print(f"{title}")
        print(f"{'='*80}{cls.STYLES['end']}")