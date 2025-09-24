import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class ThemePage extends StatefulWidget {
  const ThemePage({super.key});

  static const route = '/theme';

  @override
  State<ThemePage> createState() => _ThemePageState();
}

class _ThemePageState extends State<ThemePage> {
  String currentTheme = 'Light';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _header(context, 'Theme'),
            const SizedBox(height: 24),
            _themeOption('Light', Icons.wb_sunny_outlined),
            _themeOption('Dark', Icons.nights_stay_outlined),
            _themeOption('System default', Icons.settings_suggest_outlined),
            const SizedBox(height: 24),
            ElevatedButton(onPressed: () {}, child: const Text('Apply Theme')),
          ],
        ),
      ),
    );
  }

  Widget _themeOption(String label, IconData icon) {
    final selected = currentTheme == label;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () => setState(() => currentTheme = label),
        borderRadius: BorderRadius.circular(18),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: selected ? AppColors.primaryBlue : Colors.transparent,
              width: 2,
            ),
          ),
          child: Row(
            children: [
              Icon(icon, color: AppColors.primaryBlue),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  label,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                  ),
                ),
              ),
              if (selected)
                const Icon(Icons.check_circle, color: AppColors.primaryBlue),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header(BuildContext context, String title) {
    return Row(
      children: [
        IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back_ios_new),
        ),
        const Spacer(),
        Text(
          title,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const Spacer(),
        const SizedBox(width: 48),
      ],
    );
  }
}
