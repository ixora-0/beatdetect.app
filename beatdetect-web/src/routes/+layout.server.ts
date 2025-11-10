import load_config from '$lib/utils/config-loader';
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = async () => {
  const config = load_config();
  return { config };
};
